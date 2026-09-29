import io, os, math, json, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

st.set_page_config(page_title="SIH26227 | EO Change Intelligence", page_icon="🛰️", layout="wide")

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1rem;}
.hero {padding:22px 26px; border-radius:16px; background:linear-gradient(115deg,#07131f,#123c55); color:white; margin-bottom:18px;}
.hero h1 {margin:0; font-size:2.05rem;}
.hero p {margin:.35rem 0 0; opacity:.86;}
.card {background:white; border:1px solid #e3e8ec; border-radius:14px; padding:15px; margin-bottom:10px;}
.warn {padding:10px 13px; border-radius:10px; background:#fff7df; border:1px solid #ead38a;}
.small {font-size:.84rem; color:#66737d;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🛰️ EO Change Intelligence Workstation</h1>
<p>SIH 2026 • PS26227 • Analyst-oriented multi-temporal satellite imagery analysis</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def read_image(upload):
    data = upload.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_np(im):
    return np.array(im.convert("RGB"))

def resize_pair(a, b, max_side=1600):
    h,w = a.shape[:2]
    scale = min(1.0, max_side/max(h,w))
    nw, nh = max(1,int(w*scale)), max(1,int(h*scale))
    a2 = cv2.resize(a,(nw,nh),interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b,(nw,nh),interpolation=cv2.INTER_AREA)
    return a2,b2

def align_ecc(a,b):
    # Align B to A. Good for small translation/rotation differences.
    g1=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    g2=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    warp=np.eye(2,3,dtype=np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5)
    try:
        cc,warp=cv2.findTransformECC(g1,g2,warp,cv2.MOTION_AFFINE,criteria,None,5)
        out=cv2.warpAffine(b,warp,(a.shape[1],a.shape[0]),
                           flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)
        return out, float(cc), warp
    except Exception:
        return b, 0.0, warp

def quality(im):
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    mean=float(gray.mean())
    return lap,mean

def edge_map(im):
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    return cv2.Canny(gray,60,160)

def change_map(a,b,method="Hybrid",threshold=0.22):
    aa=a.astype(np.float32)/255
    bb=b.astype(np.float32)/255
    if method=="RGB Difference":
        d=np.mean(np.abs(aa-bb),axis=2)
    elif method=="Edge Difference":
        ea=edge_map(a).astype(np.float32)/255
        eb=edge_map(b).astype(np.float32)/255
        d=np.abs(ea-eb)
    elif method=="Normalized Difference":
        # channel-wise normalized difference; useful for RGB proxies, not a true spectral index.
        d=np.mean(np.abs((aa-bb)/(aa+bb+1e-5)),axis=2)
        d=np.clip(d,0,1)
    else:
        rgb=np.mean(np.abs(aa-bb),axis=2)
        ea=edge_map(a).astype(np.float32)/255
        eb=edge_map(b).astype(np.float32)/255
        ed=np.abs(ea-eb)
        d=0.65*rgb+0.35*ed
    # suppress tiny speckle
    mask=(d>=threshold).astype(np.uint8)*255
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return d,mask

def boxes_from_mask(mask,min_area=120):
    n, labels, stats, cents=cv2.connectedComponentsWithStats(mask,8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<min_area: continue
        rows.append((int(x),int(y),int(w),int(h),int(area)))
    rows.sort(key=lambda x:x[4], reverse=True)
    return rows

def draw_boxes(img, boxes, max_boxes=100):
    out=img.copy()
    for j,(x,y,w,h,area) in enumerate(boxes[:max_boxes],1):
        cv2.rectangle(out,(x,y),(x+w,y+h),(255,40,40),2)
        cv2.putText(out,f"CHG-{j}",(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,40,40),1,cv2.LINE_AA)
    return out

def heatmap(d):
    x=np.uint8(np.clip(d*255,0,255))
    return cv2.applyColorMap(x,cv2.COLORMAP_TURBO)[:,:,::-1]

def blend(a, overlay, alpha=.45):
    return cv2.addWeighted(a,1-alpha,overlay,alpha,0)

def model_status():
    p=Path("models/changeformer_levir.pt")
    return p.exists(), p

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Analyst Controls")
    st.caption("Prototype • no classified imagery or operational claims")
    method=st.selectbox("Change analysis",["Hybrid","RGB Difference","Edge Difference","Normalized Difference"])
    threshold=st.slider("Change threshold",0.05,0.80,0.22,0.01)
    min_area=st.slider("Minimum change object area",20,5000,120,10)
    do_align=st.checkbox("Co-register / align T2 to T1",True)
    st.divider()
    st.subheader("Evidence layers")
    show_edges=st.checkbox("Edge comparison",True)
    show_boxes=st.checkbox("Change boxes",True)
    show_heat=st.checkbox("Change heatmap",True)
    st.divider()
    st.subheader("AI model")
    cf_ok,_=model_status()
    if cf_ok:
        st.success("ChangeFormer checkpoint detected")
    else:
        st.info("ChangeFormer weights not installed")
        st.caption("Use models/download_models.py or place the official LEVIR checkpoint at models/changeformer_levir.pt.")

# -----------------------------
# Upload
# -----------------------------
st.subheader("1. Temporal imagery")
c1,c2,c3=st.columns([1,1,1])
with c1:
    t1=st.file_uploader("T1 — previous / baseline image",type=["png","jpg","jpeg","tif","tiff"],key="t1")
with c2:
    t2=st.file_uploader("T2 — current image",type=["png","jpg","jpeg","tif","tiff"],key="t2")
with c3:
    st.markdown('<div class="card"><b>Supported prototype evidence</b><br>RGB imagery • same AOI • preferably same season/sensor • accurate co-registration</div>',unsafe_allow_html=True)

if not (t1 and t2):
    st.info("Upload both T1 and T2 images to start temporal change analysis.")
    st.stop()

A=pil_np(read_image(t1))
B=pil_np(read_image(t2))
A,B=resize_pair(A,B)

st.subheader("2. Registration & image quality")
if do_align:
    B_aligned,ecc,warp=align_ecc(A,B)
else:
    B_aligned=B.copy(); ecc=0.0; warp=np.eye(2,3)

qa,ma=quality(A)
qb,mb=quality(B_aligned)
m1,m2,m3,m4=st.columns(4)
m1.metric("T1 sharpness",f"{qa:.1f}")
m2.metric("T2 sharpness",f"{qb:.1f}")
m3.metric("Alignment score",f"{ecc:.3f}" if do_align else "OFF")
m4.metric("Image size",f"{A.shape[1]} × {A.shape[0]}")

if do_align and ecc < 0.70:
    st.warning("Low registration confidence. Change detections may contain false positives caused by mis-registration.")

# -----------------------------
# Analysis
# -----------------------------
d,mask=change_map(A,B_aligned,method,threshold)
boxes=boxes_from_mask(mask,min_area)

st.subheader("3. Multi-view change analysis")
tab1,tab2,tab3,tab4,tab5=st.tabs(["Overview","Edges","Change Objects","AI ChangeFormer","Analyst Review"])

with tab1:
    x,y,z=st.columns(3)
    x.image(A,caption="T1 baseline",use_container_width=True)
    y.image(B_aligned,caption="T2 aligned/current",use_container_width=True)
    z.image(blend(A,heatmap(d)),caption="Change intensity overlay",use_container_width=True)
    pct=float((mask>0).mean()*100)
    a,b,c=st.columns(3)
    a.metric("Changed pixels",f"{pct:.2f}%")
    b.metric("Change objects",len(boxes))
    c.metric("Largest object",f"{boxes[0][4]:,} px" if boxes else "0")

with tab2:
    ea=edge_map(A); eb=edge_map(B_aligned)
    ecomp=np.maximum(ea,eb)
    ec=cv2.absdiff(ea,eb)
    x,y,z=st.columns(3)
    x.image(ea,caption="T1 edges",use_container_width=True)
    y.image(eb,caption="T2 edges",use_container_width=True)
    z.image(ec,caption="Edge difference",use_container_width=True)
    st.caption("Edge difference is an evidence layer for boundaries/structures; it is not by itself a semantic change detector.")

with tab3:
    boxed=draw_boxes(B_aligned,boxes)
    st.image(boxed,caption=f"Detected change objects: {len(boxes)}",use_container_width=True)
    if boxes:
        df=pd.DataFrame([{"ID":i+1,"x":x,"y":y,"width":w,"height":h,"area_px":area,
                         "priority":"HIGH" if area>5000 else "MEDIUM" if area>1000 else "LOW"}
                        for i,(x,y,w,h,area) in enumerate(boxes)])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download change-object CSV",df.to_csv(index=False).encode(),"change_objects.csv","text/csv")

with tab4:
    ok,path=model_status()
    if not ok:
        st.warning("Official ChangeFormer weights are not bundled because the checkpoint is a separately distributed binary. Download it using models/download_models.py.")
        st.markdown("**Model:** ChangeFormerV6 trained on LEVIR-CD. It is a building-change benchmark model, so its output should be treated as evidence rather than an operational military-change label.")
        st.code("python models/download_models.py",language="powershell")
    else:
        st.info("Checkpoint found. The adapter is intentionally isolated in `models/changeformer_adapter.py` so the official model implementation can be updated independently.")
        st.warning("This demo currently keeps the deterministic CV layers as the guaranteed path. Connect the official ChangeFormer implementation/checkpoint through the adapter before claiming model-level accuracy.")

with tab5:
    st.markdown("### Analyst decision")
    st.caption("The system produces evidence; the analyst makes the final interpretation.")
    decision=st.radio("Review status",["Unreviewed","Likely change","Likely false alarm","Needs more imagery"],horizontal=True)
    notes=st.text_area("Analyst notes",placeholder="Record evidence, uncertainty, registration issues, seasonal effects, etc.")
    if st.button("Create analyst evidence record"):
        record={
            "timestamp_utc":datetime.utcnow().isoformat()+"Z",
            "t1_file":t1.name,"t2_file":t2.name,
            "method":method,"threshold":threshold,"min_area":min_area,
            "alignment_score":ecc,"changed_pixels_pct":pct,
            "change_objects":len(boxes),"decision":decision,"notes":notes
        }
        st.download_button("Download JSON audit record",json.dumps(record,indent=2).encode(),"analyst_review.json","application/json")

# -----------------------------
# False alarm guidance
# -----------------------------
st.subheader("4. False-alarm controls")
st.markdown("""
- **Registration check:** warns when temporal alignment is weak.
- **Image-quality check:** exposes sharpness/blur differences.
- **Multi-view evidence:** RGB difference + edge difference + object grouping.
- **Analyst review:** every detected object can be reviewed rather than automatically treated as a real-world event.
- **Temporal caution:** seasonal vegetation, clouds, haze, shadows and sensor/viewing differences can create false changes.
""")

st.caption("Prototype status: research/demo workstation. It does not establish classified intelligence, ownership, military activity, legal violations, or operational truth from imagery.")

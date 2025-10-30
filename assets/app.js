// ---------- API base (same-origin by default) ----------
const BASE =
  (typeof window !== "undefined" && typeof window.API_BASE === "string")
    ? window.API_BASE
    : ""; // same-origin fallback

export const URLS = {
  brain: `${BASE}/api/brain/predict`,
  lung:  `${BASE}/api/lung/predict`,
};

// Example usage from your existing code:
// const fd = new FormData(); fd.append("image", file);
// const res = await fetch(URLS.lung, { method: "POST", body: fd });

const DEFAULT_LABELS={
  brain:["glioma","meningioma","no_tumor","pituitary"],
  lung:["Bacterial Pneumonia","Corona Virus Disease","Normal","Tuberculosis","Viral Pneumonia"]
};
window.LABELS={...DEFAULT_LABELS};

const EDU={
  brain:{
    glioma:{what:"Gliomas are brain tumors arising from glial cells. Severity varies by grade.",
      why:"They grow from support cells in the brain; exact causes are often unclear.",
      next:"Neurosurgical review; treatment may include surgery and radiotherapy ± chemotherapy."},
    meningioma:{what:"Meningiomas grow from the brain’s coverings and are often slow-growing/benign.",
      why:"Sometimes incidental; risks include prior radiation or certain genetic factors.",
      next:"Small asymptomatic tumors may be monitored; otherwise surgery or radiotherapy is considered."},
    pituitary:{what:"Usually benign growths near the base of the brain; some change hormones.",
      why:"May be hormone-secreting or non-functioning; often found on imaging.",
      next:"Care via endocrinology/neurosurgery: medicines, endoscopic surgery, and/or radiotherapy."},
    no_tumor:{what:"No tumor features detected by the model.",
      why:"AI may miss subtle findings; imaging quality and other conditions matter.",
      next:"Use clinical judgment; if symptoms persist, seek medical evaluation."}
  },
  lung:{
    "Bacterial Pneumonia":{what:"A lung infection caused by bacteria with fever, cough, breathlessness.",
      why:"Air sacs fill with inflammatory fluid; risks include age and chronic illness.",
      next:"Clinician may prescribe antibiotics; vaccines and timely care reduce complications."},
    "Corona Virus Disease":{what:"COVID-19 respiratory illness caused by SARS-CoV-2.",
      why:"Spreads via droplets/aerosols; higher risk with comorbidities.",
      next:"Test and follow isolation guidance; antivirals may be used for higher-risk patients per current practice."},
    "Normal":{what:"No clear abnormality detected by the model.",
      why:"‘Normal’ on AI doesn’t rule out early/subtle disease.",
      next:"Correlate with symptoms; follow up if concerns persist."},
    "Tuberculosis":{what:"Contagious lung infection from Mycobacterium tuberculosis.",
      why:"Airborne spread; reactivation risk rises with immune compromise.",
      next:"Needs confirmed testing and multi-drug therapy under public health supervision."},
    "Viral Pneumonia":{what:"Lung infection from viruses such as influenza or RSV.",
      why:"Viruses inflame airways and air sacs; severe in infants/older adults/comorbidity.",
      next:"Care is usually supportive; specific antivirals depend on the virus and clinician judgment."}
  }
};

document.addEventListener("DOMContentLoaded",async()=>{
  const y=document.getElementById("y"); if(y) y.textContent=new Date().getFullYear();
  try{
    const h=await fetch(`${window.API_BASE}/health`).then(r=>r.json());
    if(h?.brain?.labels?.length) window.LABELS.brain=h.brain.labels;
    if(h?.lung?.labels?.length && !/^class_\d+/i.test(h.lung.labels[0]||"")) window.LABELS.lung=h.lung.labels;
    const status=document.getElementById("status");
    if(status){ status.textContent=`brain=${h?.brain?.model||"—"}, lung=${h?.lung?.model||"—"}`; }
  }catch{}

  document.querySelectorAll(".dropzone").forEach(wireDropzone);
  document.querySelectorAll("[data-analyze]").forEach(btn=>btn.addEventListener("click",()=>analyze(btn.dataset.analyze)));
});

function wireDropzone(dz){
  const zone=dz.dataset.zone;
  const input=dz.querySelector('input[type="file"]');
  const info=document.querySelector(`.dz-info[data-info="${zone}"]`);
  const preview=document.getElementById(`preview-${zone}`);
  const setInfo=f=>{
    if(!f){ info.textContent="No file selected."; if(preview) preview.style.display="none"; return; }
    info.textContent=`Selected: ${f.name} • ${humanSize(f.size)}`;
    if(preview){ const r=new FileReader(); r.onload=e=>{preview.src=e.target.result; preview.style.display="block"}; r.readAsDataURL(f); }
  };
  input.addEventListener("change",e=>setInfo(e.target.files[0]));
  ["dragenter","dragover"].forEach(evt=>dz.addEventListener(evt,e=>{e.preventDefault();dz.classList.add("drag");}));
  ["dragleave","drop"].forEach(evt=>dz.addEventListener(evt,e=>{e.preventDefault();dz.classList.remove("drag");}));
  dz.addEventListener("drop",e=>{const f=e.dataTransfer.files&&e.dataTransfer.files[0]; if(f){input.files=e.dataTransfer.files; setInfo(f);}});
}
function humanSize(b){const u=["B","KB","MB","GB"];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++;}return (i?b.toFixed(1):b)+" "+u[i];}
function labelFrom(zone,k){return String(k).startsWith("class_")?(window.LABELS[zone][parseInt(k.split("_")[1],10)]??k):k;}

async function analyze(zone){
  const input=document.getElementById(`${zone}-file`);
  const result=document.getElementById(`${zone}-result`);
  const bar=document.getElementById(`${zone}-bar`);
  const btn=document.querySelector(`[data-analyze="${zone}"]`);
  const info=document.querySelector(`.dz-info[data-info="${zone}"]`);
  if(!input||!input.files||!input.files[0]){ if(info) info.textContent="Please choose an image first."; return; }

  const original=btn?btn.textContent:""; if(btn){btn.disabled=true;btn.textContent="Analyzing…";} if(bar) bar.style.width="18%";
  result.innerHTML="Running inference…";

  try{
    const fd=new FormData(); fd.append("image",input.files[0]);
    const res=await fetch(`${window.API_BASE}/api/${zone}/predict`,{method:"POST",body:fd});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data=await res.json();

    const top=labelFrom(zone,data.top); const confPct=data.conf!=null?(data.conf*100).toFixed(1)+"%":"—";
    const edu=(EDU[zone]&&EDU[zone][top])?EDU[zone][top]:null;

    result.innerHTML=`
      <div class="verdict"><strong>Prediction:</strong> ${top} <span class="muted">(${confPct})</span></div>
      <div class="chips">
        ${zone==="brain"
          ? `<span class="chip"><i>🧠</i>MRI</span><span class="chip"><i>📐</i>~299×299</span><span class="chip"><i>ℹ️</i>Assist tool</span>`
          : `<span class="chip"><i>🫁</i>Chest X-ray</span><span class="chip"><i>📐</i>~224×224</span><span class="chip"><i>ℹ️</i>Assist tool</span>`
        }
      </div>
      <div class="edu card" style="margin-top:10px">
        <h3 style="margin-bottom:6px">${top}: what to know</h3>
        <p>${edu ? edu.what : "Information not available."}</p>
        <p><strong>Why it happens:</strong> ${edu ? edu.why : "—"}</p>
        <p><strong>What to do next:</strong> ${edu ? edu.next : "Consult a clinician."}</p>
        <p class="muted" style="margin-top:8px">Educational content only. Not a diagnosis or treatment plan.</p>
      </div>`;
  }catch(e){
    result.innerHTML=`<span style="color:#fca5a5">Error:</span> ${e.message}`;
  }finally{
    if(bar) bar.style.width="100%"; setTimeout(()=>{if(bar) bar.style.width="0%";},600);
    if(btn){btn.disabled=false;btn.textContent=original;}
  }
}

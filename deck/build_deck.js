const P=require("pptxgenjs");
const pres=new P(); pres.defineLayout({name:"W",width:13.333,height:7.5}); pres.layout="W";
const PAPER="FFFFFF",INK="1A1A1A",MUTE="6E6E6E",RULE="DEDCD7",TINT="F6F5F2",
      ACC="8C3A1E",GRN="1B7A4B";
const HEAD="Cambria",BODY="Calibri";
const W=13.333,H=7.5,M=0.85,CW=W-2*M;
function slide(k,t,s){const x=pres.addSlide();x.background={color:PAPER};
 x.addText(k.toUpperCase(),{x:M,y:0.38,w:CW,h:0.22,fontFace:BODY,fontSize:10,bold:true,color:ACC,charSpacing:1.6,margin:0});
 x.addText(t,{x:M,y:0.64,w:CW,h:0.54,fontFace:HEAD,fontSize:26,bold:true,color:INK,margin:0});
 if(s)x.addText(s,{x:M,y:1.22,w:CW,h:0.30,fontFace:BODY,fontSize:12,color:MUTE,italics:true,margin:0});
 x.addShape(pres.ShapeType.rect,{x:M,y:s?1.58:1.30,w:CW,h:0.012,fill:{color:RULE}});return x;}
function tint(s,x,y,w,h,c){s.addShape(pres.ShapeType.rect,{x,y,w,h,fill:{color:c||TINT}});}
function lab(s,x,y,w,t,o){o=o||{};s.addText(t,{x,y,w,h:0.22,fontFace:BODY,fontSize:o.size||10.5,bold:true,color:o.color||ACC,charSpacing:0.8,margin:0});}
function tbl(s,rows,opt){opt=opt||{};const ctr=opt.centerCols||[],hi=opt.hiRows||[];
 const sh=rows.map((r,ri)=>r.map((c,ci)=>{const o={};
  if(ri===0){o.bold=true;o.fill={color:TINT};}
  if(hi.indexOf(ri)!==-1){o.bold=true;o.fill={color:"EAF5EE"};}
  if(ctr.indexOf(ci)!==-1)o.align="center";return {text:String(c),options:o};}));
 const t=Object.assign({x:M,w:CW,border:{type:"solid",color:RULE,pt:0.5},fontFace:BODY,
  fontSize:10,color:INK,valign:"top",margin:4},opt);
 delete t.centerCols;delete t.hiRows;s.addTable(sh,t);}

/* 1 TITLE */
{const s=pres.addSlide();s.background={color:PAPER};tint(s,0,0,W,2.0,TINT);
 s.addText("Foundations of Data Science  ·  Review 1",{x:M,y:0.56,w:CW,h:0.26,fontFace:BODY,fontSize:11.5,bold:true,color:ACC,charSpacing:1.6,margin:0});
 s.addText("Instruction-Guided Edit-Region Prediction",{x:M,y:0.90,w:CW,h:0.58,fontFace:HEAD,fontSize:32,bold:true,color:INK,margin:0});
 s.addText("Predicting where an instruction says to edit, at 1% of the parameters",{x:M,y:2.16,w:CW,h:0.36,fontFace:BODY,fontSize:14,color:MUTE,italics:true,margin:0});
 s.addShape(pres.ShapeType.rect,{x:M,y:2.68,w:3.2,h:0.014,fill:{color:ACC}});
 s.addText([{text:"Amritha S",options:{bold:true}},{text:"   23BEC1368"}],{x:M,y:2.96,w:5.6,h:0.3,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText([{text:"Yugeshwaran P",options:{bold:true}},{text:"   23BEC1404"}],{x:M,y:3.32,w:5.6,h:0.3,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText("Guide  ·  Dr. Saranya M  (54783)\nSchool of Electronics Engineering  ·  VIT Chennai",{x:M,y:3.88,w:6.2,h:0.7,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
 tint(s,W-M-4.8,2.90,4.8,2.05,"EAF5EE");
 lab(s,W-M-4.55,3.10,4.3,"Headline result",{color:GRN});
 s.addText([{text:"IoU 0.1718",options:{bold:true}},{text:"  ·  beats the human annotations the base paper trains on (0.1511)\n"},
            {text:"+71%",options:{bold:true}},{text:"  on insertion vs CLIPSeg\n"},
            {text:"9.7×",options:{bold:true}},{text:"  faster, "},{text:"108×",options:{bold:true}},{text:"  fewer trainable parameters"}],
   {x:W-M-4.55,y:3.38,w:4.3,h:1.4,fontFace:BODY,fontSize:11.5,color:GRN,lineSpacing:17,margin:0});
 s.addShape(pres.ShapeType.rect,{x:0,y:H-0.5,w:W,h:0.012,fill:{color:RULE}});
 s.addNotes("PH1");}

/* 2 PROBLEM */
{const s=slide("The problem","An editor must decide where to edit before it can edit","Everyone works on the drawing. Almost nobody measures the decision.");
 const r=[["Instruction","What must change","An existing object?"],
  ["make the jacket blue","the jacket, not the person","No — a part"],
  ["remove the cat on the left","one instance, by position","Yes, but which?"],
  ["put a hat on the man","empty space above his head","No — nothing is there"]];
 tbl(s,r,{y:1.88,colW:[3.6,4.6,3.43],rowH:0.48,fontSize:11.5});
 tint(s,M,4.15,CW,0.92);
 s.addText("Referring segmentation returns what a phrase denotes. For “put a hat on the man” the phrase denotes the man — the region that must change is above his head.",
  {x:M+0.32,y:4.34,w:CW-0.64,h:0.58,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 lab(s,M,5.30,9,"Why this is a data-science problem");
 s.addText("No benchmark scores a predicted edit region. We checked the base paper directly: the word “IoU” appears once, qualitatively. A 2026 paper whose headline contribution is a MaskPredictor never measures how well it localizes.",
  {x:M,y:5.58,w:CW,h:0.7,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 s.addNotes("PH2");}

/* 3 BASE PAPER */
{const s=slide("Base paper","AdaptEdit — Edit Where You Mean  (arXiv:2604.23763, April 2026)","The only 2026 work that trains a predictor to ground the edit region from the instruction.");
 const a=[["§","Component","Params"],
  ["3.1","Qwen-Image-Edit DiT, 60 blocks — frozen","20.43 B"],
  ["3.2","Block Adapters ×60","4.67 B"],
  ["3.3","Condition Encoder","351 M"],
  ["3.4","SpatialGate","—"],
  ["3.6","MaskPredictor — FiLM query-grid decoder","7.8 M"]];
 tbl(s,a,{y:1.88,w:6.4,colW:[0.5,4.3,1.6],rowH:0.34,fontSize:10,hiRows:[5]});
 tint(s,M+6.8,1.88,CW-6.8,2.25,"FBEAEA");
 lab(s,M+7.05,2.08,4.3,"Its own stated limitation",{color:"A33A2C"});
 s.addText("“Geometry-changing edits with no localized source region (‘put a hat on the dog’ where the dog has no hat): the GT mask covers the new content’s location but there is no clean source signal to ‘protect’ outside it.”",
  {x:M+7.05,y:2.38,w:4.3,h:1.65,fontFace:HEAD,fontSize:11,italics:true,color:"A33A2C",lineSpacing:15,margin:0});
 lab(s,M,4.05,10,"What we can and cannot do with it");
 const b=[["Cannot","run or retrain it","20.43 B frozen + 5.0 B trainable needs multi-GPU"],
  ["Cannot","reproduce its numbers","it reports L1 / CLIP-I / DINO on edited images, never a mask IoU"],
  ["Can","adopt its formulation","predict the region from the instruction; compare on its data"],
  ["Can","reimplement its mechanism","the MaskPredictor head, at our scale, under matched conditions"]];
 tbl(s,b,{y:4.33,colW:[1.0,2.5,8.13],rowH:0.34,fontSize:10});
 s.addNotes("PH3");}

/* 4 SUPERVISION */
{const s=slide("Finding 1","The supervision the field trains on is 9.1× too coarse","All 528 MagicBrush dev turns, against the region that actually changed.");
 const r=[["edit type","n","human mask","actually changed","ratio","recall","precision","IoU"],
  ["insert","103","60.0%","7.0%","11.9×","97.8%","11.9%","0.12"],
  ["modify","390","62.9%","11.2%","8.5×","96.5%","17.1%","0.17"],
  ["remove","35","63.6%","8.9%","7.1×","96.2%","13.4%","0.13"],
  ["all","528","62.4%","10.3%","9.1×","96.7%","15.8%","0.16"]];
 tbl(s,r,{y:1.88,colW:[1.5,0.8,1.75,2.0,1.1,1.3,1.35,1.83],rowH:0.38,fontSize:10.5,centerCols:[1,2,3,4,5,6,7],hiRows:[4]});
 tint(s,M,4.10,CW,1.05);
 s.addText("Recall 96.7% — the annotations are correct.  Precision 15.8% — only a sixth of the annotated region actually changes. These are region-of-interest scribbles for an inpainting tool, not edit regions.",
  {x:M+0.32,y:4.32,w:CW-0.64,h:0.7,fontFace:BODY,fontSize:12,color:INK,lineSpacing:17,margin:0});
 lab(s,M,5.35,9,"So the supervision is the opportunity");
 s.addText("We keep MagicBrush’s real human before/after pairs but derive the target ourselves by CIE76 ΔE in L*a*b*. Where an object actually landed is a far tighter signal than where a human roughly brushed.",
  {x:M,y:5.63,w:CW,h:0.65,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
 s.addNotes("PH4");}

/* 5 ARCHITECTURE */
{const s=slide("Our architecture","One mechanism changed, and it is significant","Frozen 71.75 M backbone + frozen CLIP + a 1.4 M trainable head.");
 lab(s,M,1.88,6,"Theirs — decode from learned queries");
 s.addText("M = σ( Upsample( ConvHead( CA(q = γ⊙Q + β,  k,v = φ(features)) ) ) )",
  {x:M,y:2.14,w:6.0,h:0.32,fontFace:"Consolas",fontSize:10.5,color:INK,margin:0});
 s.addText("A learned 16×16 query grid cross-attends to patchified features. Spatial structure is learned from scratch.",
  {x:M,y:2.50,w:6.0,h:0.5,fontFace:BODY,fontSize:11,color:MUTE,lineSpacing:15,margin:0});
 lab(s,M,3.12,6,"Ours — a coefficient field over an existing basis",{color:GRN});
 s.addText("M = σ( Σ cᵢ(x,y) · protoᵢ(x,y) + b )",{x:M,y:3.38,w:6.0,h:0.34,fontFace:"Consolas",fontSize:12.5,bold:true,color:GRN,margin:0});
 s.addText("FastSAM already emits 32 prototype masks. Their coefficients normally come from a detection — which is why its output can only be an existing object. The instruction predicts them instead, as a 20×20 spatial field, so different regions use different mixtures.",
  {x:M,y:3.76,w:6.0,h:1.0,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 tint(s,M+6.5,1.88,CW-6.5,2.35,"EAF5EE");
 lab(s,M+6.75,2.08,4.5,"Measured head-to-head, matched conditions",{color:GRN});
 const h=[["head","params","IoU"],
  ["their mechanism","923 K","0.1050"],
  ["ours, global coeff","1.4 M","0.1590"],
  ["ours, spatial field","4.7 M","0.1718"]];
 tbl(s,h,{x:M+6.75,y:2.38,w:4.4,colW:[2.0,1.1,1.3],rowH:0.32,fontSize:10,centerCols:[1,2],hiRows:[3]});
 s.addText("Identical inputs, data, loss, schedule and seeds.\nSpatial vs global:  p = 0.0070  (12 seeds).",
  {x:M+6.75,y:3.78,w:4.4,h:0.5,fontFace:BODY,fontSize:10.5,color:GRN,lineSpacing:14,margin:0});
 tint(s,M+6.5,4.45,CW-6.5,1.15,"FBEAEA");
 lab(s,M+6.75,4.62,4.5,"What is shared, not claimed",{color:"A33A2C"});
 s.addText("FiLM modulation and BCE + Dice are their design too — we reached the same choices independently.",
  {x:M+6.75,y:4.88,w:4.4,h:0.6,fontFace:BODY,fontSize:10.5,color:"A33A2C",lineSpacing:14,margin:0});
 s.addNotes("PH5");}

/* 6 RESULTS FIGURE */
{const s=slide("Results","Accuracy against cost, and where the win is","503 held-out dev samples · trained on 2,278 disjoint samples · 12 seeds");
 s.addImage({path:"img/frontier.png",x:M,y:1.86,w:CW,h:CW/2.81});
 tint(s,M,6.10,CW,0.95,"EAF5EE");
 s.addText([{text:"0.1718 ± 0.0099",options:{bold:true}},{text:"  —  above the human masks (0.1511) and 93% of CLIPSeg (0.1849), at "},
  {text:"108× fewer trainable parameters",options:{bold:true}},{text:" and "},{text:"9.7× lower latency",options:{bold:true}},
  {text:".  On insertion we beat CLIPSeg by 71%; the human masks still lead there."}],
  {x:M+0.32,y:6.30,w:CW-0.64,h:0.62,fontFace:BODY,fontSize:11.5,color:GRN,lineSpacing:16,margin:0});
 s.addNotes("PH6");}

/* 7 ANALYSIS FIGURE */
{const s=slide("Analysis","Cost, controls, and how much headroom is left",null);
 s.addImage({path:"img/analysis.png",x:M,y:1.55,w:CW,h:CW/3.34});
 const r=[["Cost amortises","break-even at N = 2 instructions on one image; 9.0× at N = 100, because the backbone runs once and each instruction costs 5.6 ms"],
  ["Controls included","the spatial field is significant (p = 0.0070); the geometric basis is not (p = 0.33 against a random control) and was dropped"],
  ["Headroom is known","we reach 44% / 50% / 42% of the least-squares ceiling — the bound on any coefficient predictor over this basis"]];
 tbl(s,r,{y:5.30,colW:[2.6,9.03],rowH:0.42,fontSize:10});
 s.addNotes("PH7");}

/* 8 HONEST LIMITS */
{const s=slide("What we checked on ourselves","Every number here survived an attempt to break it",null);
 const r=[["Check","What it caught"],
  ["Contamination audit","an early headline trained on the evaluation split; retracted in the repository and re-run on disjoint shards"],
  ["Selection-bias audit","choosing the best epoch by dev IoU had inverted the v1/v2 conclusion; selection moved to a held-out slice of train"],
  ["Random-basis controls","two proposed improvements — free-space and geometric bases — failed against noise and were dropped"],
  ["Resolution matching","latency was measured at 640 px against CLIPSeg’s 352 px; re-measured matched, 9.7× rather than 10.8×"],
  ["Ceiling consistency","the headroom figure compared a train-set bound to a dev score; recomputed on dev"]];
 tbl(s,r,{y:1.42,colW:[2.9,8.73],rowH:0.62,fontSize:10.5});
 tint(s,M,5.10,CW,0.95);
 s.addText("Two claims were retracted and two proposed improvements rejected by their own controls. That is why the three results that remain are worth stating.",
  {x:M+0.32,y:5.32,w:CW-0.64,h:0.6,fontFace:HEAD,fontSize:13,italics:true,color:INK,lineSpacing:17,margin:0});
 lab(s,M,6.25,10,"Known limitation, not yet quantified");
 s.addText("The remove ceiling on dev rests on 32 samples and carries uncertainty the figure does not show.",
  {x:M,y:6.50,w:CW,h:0.4,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,margin:0});
 s.addNotes("PH8");}

/* 9 PLAN */
{const s=slide("What follows","Three months, and the bar each step must clear",null);
 const p=[["Month","Focus","Deliverable","Must beat"],
  ["1 · Sep–Oct","Supervision at scale","all 51 MagicBrush shards (~8,800 turns); re-measure the scaling curve","0.1718 at 2,278 samples"],
  ["2 · Oct–Nov","Features, not heads","exp08/11 show the head is not the insertion bottleneck — test text-conditioned features","a random-basis control of equal size"],
  ["3 · Nov–Dec","Editing and evaluation","mask-conditioned inpainting; collateral change outside the region — Review 2","CLIPSeg at 0.1849"]];
 tbl(s,p,{y:1.38,colW:[1.6,2.1,5.4,2.53],rowH:0.70,fontSize:10});
 lab(s,M,4.15,9,"The methodological bar this project already holds itself to");
 const b=[["Every claim reports its ablation","a config scoring IoU 0.1429 with delta −0.0012 was a blob predictor ignoring the instruction"],
  ["Every fix must beat its control","free-space and geometric bases both lost to random noise and were dropped"],
  ["Every split is disjoint by construction","train and dev come from different shard files, never merged"],
  ["Every figure regenerates from the data","python train/figures.py reads only the committed result JSONs"]];
 tbl(s,b,{y:4.43,colW:[3.9,7.73],rowH:0.40,fontSize:10});
 s.addNotes("PH9");}

/* 10 THANK YOU */
{const s=pres.addSlide();s.background={color:PAPER};tint(s,0,0,W,2.4,TINT);
 s.addText("Thank you",{x:M,y:0.78,w:CW,h:0.8,fontFace:HEAD,fontSize:38,bold:true,color:INK,margin:0});
 s.addText("We are happy to take questions.",{x:M,y:1.64,w:CW,h:0.4,fontFace:BODY,fontSize:15,color:MUTE,italics:true,margin:0});
 s.addShape(pres.ShapeType.rect,{x:M,y:2.88,w:3.2,h:0.014,fill:{color:ACC}});
 s.addText([{text:"Amritha S",options:{bold:true}},{text:"   23BEC1368"}],{x:M,y:3.16,w:5.5,h:0.32,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText([{text:"Yugeshwaran P",options:{bold:true}},{text:"   23BEC1404"}],{x:M,y:3.53,w:5.5,h:0.32,fontFace:BODY,fontSize:14,color:INK,margin:0});
 s.addText("Guide  ·  Dr. Saranya M  (54783)\nSchool of Electronics Engineering  ·  VIT Chennai",{x:M,y:4.06,w:6.0,h:0.7,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
 tint(s,W-M-4.8,3.05,4.8,1.75,"EAF5EE");
 lab(s,W-M-4.55,3.25,4.3,"Everything reproduces",{color:GRN});
 s.addText("github.com/Amritha902/edit-region-prediction\n\n12 experiments  ·  seeded  ·  runs on a MacBook Air",{x:W-M-4.55,y:3.53,w:4.3,h:1.0,fontFace:BODY,fontSize:11,color:GRN,lineSpacing:15,margin:0});
 s.addNotes("PH10");}

pres.writeFile({fileName:"ISE_Review1_Final.pptx"}).then(f=>console.log("wrote",f));

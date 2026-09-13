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
 s.addText([{text:"IoU 0.2065",options:{bold:true}},{text:"  ·  beats CLIPSeg (0.1849) and the human annotations (0.1511)\n"},
            {text:"+120%",options:{bold:true}},{text:"  on insertion vs CLIPSeg\n"},
            {text:"10.3×",options:{bold:true}},{text:"  faster, "},{text:"32×",options:{bold:true}},{text:"  fewer trainable parameters"}],
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
  ["ours, global coeff","1.4 M","0.1907"],
  ["ours, spatial field","4.7 M","0.2065"]];
 tbl(s,h,{x:M+6.75,y:2.38,w:4.4,colW:[2.0,1.1,1.3],rowH:0.32,fontSize:10,centerCols:[1,2],hiRows:[3]});
 s.addText("Identical inputs, data, loss, schedule and seeds; 8,306 samples.\nSpatial vs global:  p = 5.4e-06,  Cohen's d = 2.50  (12 seeds).\nTheir mechanism was measured at 2,278 only.",
  {x:M+6.75,y:3.74,w:4.4,h:0.62,fontFace:BODY,fontSize:9.5,color:GRN,lineSpacing:13,margin:0});
 tint(s,M+6.5,4.45,CW-6.5,1.15,"FBEAEA");
 lab(s,M+6.75,4.62,4.5,"What is shared, not claimed",{color:"A33A2C"});
 s.addText("FiLM modulation and BCE + Dice are their design too — we reached the same choices independently.",
  {x:M+6.75,y:4.88,w:4.4,h:0.6,fontFace:BODY,fontSize:10.5,color:"A33A2C",lineSpacing:14,margin:0});
 s.addNotes("PH5");}

/* 6 RESULTS FIGURE */
{const s=slide("Results","Every seed, at both training-set sizes","503 held-out dev samples · 8,306 training samples · 12 seeds per configuration");
 s.addImage({path:"img/results.png",x:M,y:1.80,w:7.05,h:7.05/1.739});
 const mt=[["metric","v2 spatial","v1 global"],
  ["IoU","0.2065","0.1907"],
  ["Precision","0.3126","0.2972"],
  ["Recall","0.3878","0.3510"],
  ["F1 / Dice","0.2976","0.2757"]];
 tbl(s,mt,{x:M+7.35,y:1.92,w:4.28,colW:[1.68,1.34,1.26],rowH:0.34,fontSize:10,centerCols:[1,2],hiRows:[1]});
 tint(s,M+7.35,4.00,4.28,1.62,"EAF5EE");
 lab(s,M+7.55,4.18,3.9,"Against the baselines",{color:GRN});
 s.addText("CLIPSeg  0.1849   +0.0216  p=5.2e-07\nHuman masks  0.1511   +0.0554\nAll 12 seeds clear CLIPSeg.",
  {x:M+7.55,y:4.44,w:3.9,h:1.1,fontFace:BODY,fontSize:10.5,color:GRN,lineSpacing:15,margin:0});
 tint(s,M,6.28,CW,0.80,"EAF5EE");
 s.addText([{text:"0.2065 ± 0.0072",options:{bold:true}},{text:"  —  now above CLIPSeg (0.1849) as well as the human masks, at "},
  {text:"32× fewer trainable parameters",options:{bold:true}},{text:" and "},{text:"10.3× lower per-instruction latency",options:{bold:true}},
  {text:".  Insertion 0.1186 vs CLIPSeg 0.054 — +120%."}],
  {x:M+0.32,y:6.44,w:CW-0.64,h:0.56,fontFace:BODY,fontSize:11,color:GRN,lineSpacing:15,margin:0});
 s.addNotes("PH6");}

/* 6b KEY PARAMETERS */
{const s=slide("Configuration","Every parameter behind those numbers","Taken from the source, not from notes");
 const c=[["group","parameter","value"],
  ["Supervision","ΔE metric / threshold","CIE76 in L*a*b*  ·  12.0"],
  ["","morph kernel · min component","5×5  ·  40 px"],
  ["","sample filter (changed area)","0.4% – 45%"],
  ["Representation","input · prototypes","640×640  ·  32 at 160×160"],
  ["","text · image context","512-d CLIP  ·  640-d SPPF   (both frozen)"],
  ["Head (v2)","coefficient grid · hidden","20×20 bilinear  ·  512  ·  dropout 0"],
  ["Optimisation","AdamW · lr · decay","1e-3 cosine to 0  ·  0.01"],
  ["","batch · loss · clip","32  ·  BCE + 2×Dice  ·  norm 1.0"],
  ["Evaluation","epochs · selection","60  ·  best on train-val, never dev"],
  ["","threshold · seeds","0.5 fixed  ·  12 (1368, 1–11)"],
  ["","baseline","CIDAS/clipseg-rd64-refined"]];
 tbl(s,c,{y:1.78,colW:[2.0,3.1,6.53],rowH:0.355,fontSize:10});
 tint(s,M,6.32,CW,0.78);
 s.addText("8,306 usable training turns from all 51 MagicBrush shards (8,807 rows, 501 rejected as degenerate).  Splits verified image-disjoint: 794 dev images, 13,317 train, 0 overlap.",
  {x:M+0.32,y:6.50,w:CW-0.64,h:0.5,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("PH6b");}

/* 6c OPERATING POINT */
{const s=slide("Operating point","IoU alone hides the precision/recall trade-off","Threshold chosen on train-val, never on dev");
 const t1=[["threshold","IoU","precision","recall","predicted area"],
  ["0.2","0.2172","0.2637","0.5775","22.74%"],
  ["0.3","0.2180","0.2818","0.5063","18.10%"],
  ["0.5  (reported)","0.2065","0.3126","0.3878","11.70%"],
  ["0.7","0.1747","0.3458","0.2750","6.90%"],
  ["0.9","0.1119","0.3705","0.1479","2.74%"]];
 tbl(s,t1,{y:1.80,colW:[2.5,2.2,2.3,2.3,2.33],rowH:0.38,fontSize:10.5,centerCols:[1,2,3,4],hiRows:[3]});
 s.addText("True changed area is 9.58%, so at 0.5 the model over-predicts slightly.",
  {x:M,y:4.28,w:CW,h:0.3,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,margin:0});
 tint(s,M,4.66,CW,1.32,"EAF5EE");
 lab(s,M+0.32,4.84,8,"Selected honestly, on the train-val slice",{color:GRN});
 s.addText("v2 picks 0.3 (×7) or 0.2 (×5) → dev 0.2185 ± 0.0051, +0.0120 over fixed 0.5 (paired p = 2.9e-07). Insertion rises to 0.134, 63% of its ceiling.",
  {x:M+0.32,y:5.12,w:CW-0.64,h:0.76,fontFace:BODY,fontSize:11,color:GRN,lineSpacing:15,margin:0});
 tint(s,M,6.10,CW,0.98,"FBEAEA");
 lab(s,M+0.32,6.26,8,"Why the headline still quotes 0.5",{color:"A33A2C"});
 s.addText("CLIPSeg is evaluated at its default threshold, so a tuned comparison would tilt toward us. We report the conservative number. With both heads tuned, the v2 lead narrows from +0.0158 to +0.0093.",
  {x:M+0.32,y:6.52,w:CW-0.64,h:0.5,fontFace:BODY,fontSize:10.5,color:"A33A2C",lineSpacing:14,margin:0});
 s.addNotes("PH6c");}

/* 7 ANALYSIS FIGURE */
{const s=slide("Training dynamics","The best epoch is 2–7 of 60, and language carries the signal","3 seeds per head at 8,306 samples · ▼ marks the epoch actually selected");
 s.addImage({path:"img/curves.png",x:M,y:1.74,w:7.9,h:7.9/1.778});
 const r=[["Overfits fast","validation IoU peaks at epoch 2–7 then halves by epoch 60, while training loss falls throughout"],
  ["Selection saved it","choosing on a held-out train slice is why the reported numbers hold; the last epoch would have reported half"],
  ["Language is real","zeroing the instruction collapses IoU from 0.188 to ~0.03 — a gap of +0.158"],
  ["Reproducible","all 6 curve seeds reproduced their 12-seed dev IoU exactly, to four decimals"]];
 tbl(s,r,{x:M+8.15,y:1.86,w:3.48,colW:[1.25,2.23],rowH:0.90,fontSize:9});
 tint(s,M,6.22,CW,0.85);
 s.addText("Latency, resolution-matched at CLIPSeg's native 352 px:  v2 per-instruction 5.30 ms vs 54.3 ms = 10.3×, at 32× fewer trainable parameters.  The backbone runs once per image (33.6 ms); each further instruction costs only the head.",
  {x:M+0.32,y:6.40,w:CW-0.64,h:0.56,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addNotes("PH7");}

/* 8 STAGE 2 */
{const s=slide("Stage 2","Editing inside the predicted region — the premise, tested","60 held-out dev samples · one editor (InstructPix2Pix), only the gating mask varies");
 const r=[["gating mask","recall of the edit ↑","collateral change ↓","net ↑"],
  ["none — whole frame","33.3%","24.2%","+9.1%"],
  ["ours, as predicted","7.5%","1.5%","+6.0%"],
  ["ours, tuned — peak of the sweep","22.9%","8.8%","+14.2%"],
  ["MagicBrush human mask","31.5%","13.0%","+18.5%"],
  ["ground-truth region","30.2%","0.1%","+30.1%"]];
 tbl(s,r,{y:1.88,colW:[4.2,2.6,2.6,2.23],rowH:0.40,fontSize:10.5,centerCols:[1,2,3],hiRows:[3]});
 tint(s,M,4.50,6.1,1.55,"EAF5EE");
 lab(s,M+0.3,4.70,5.5,"The premise holds",{color:GRN});
 s.addText("Editing inside the correct region is worth 3.3× — net +30.1% against +9.1% for editing the whole frame. Our tuned mask reaches +14.2%, 1.56× better than the status quo.",
  {x:M+0.3,y:4.98,w:5.5,h:0.9,fontFace:BODY,fontSize:11,color:GRN,lineSpacing:15,margin:0});
 tint(s,M+6.4,4.50,CW-6.4,1.55);
 lab(s,M+6.7,4.70,4.5,"And the diagnosis is precise");
 s.addText("Our mask is precision-biased — 1.5% collateral but only 7.5% recall, so gating discarded the edit. Dilation trades precision for recall; swept to 200 px, net peaks at 64 px and falls away either side.",
  {x:M+6.7,y:4.98,w:4.3,h:0.9,fontFace:BODY,fontSize:11,color:INK,lineSpacing:15,margin:0});
 s.addText("Coarse supervision costs at the editing stage too, not only on the mask metric: the MagicBrush mask — measured 9.1× too large — loses 11.6 points of net against the ground-truth region.",
  {x:M,y:6.22,w:CW,h:0.5,fontFace:BODY,fontSize:10.5,color:MUTE,italics:true,lineSpacing:14,margin:0});
 s.addNotes("PH8b");}

/* 8 HONEST LIMITS */
{const s=slide("What we checked on ourselves","Every number here survived an attempt to break it",null);
 const r=[["Check","What it caught"],
  ["Contamination audit","an early headline trained on the evaluation split; retracted in the repository and re-run on disjoint shards"],
  ["Selection-bias audit","choosing the best epoch by dev IoU had inverted the v1/v2 conclusion; selection moved to a held-out slice of train"],
  ["Random-basis controls","two proposed improvements — free-space and geometric bases — failed against noise and were dropped"],
  ["Latency fairness","measured at 640 px against CLIPSeg’s 352 px, and only for v1 — re-measured matched and for the reported head: v2 is 10.3×"],
  ["Ceiling consistency","the headroom figure compared a train-set bound to a dev score; recomputed on dev"],
  ["Leakage audit","MagicBrush is multi-turn, so before scaling 3.65× we hashed every image on both sides — 0 of 794 dev images appear in train"],
  ["Threshold selection","dev peaks at 0.3, but adopting it would be test-set tuning; the threshold is chosen on train-val instead"]];
 tbl(s,r,{y:1.42,colW:[2.9,8.73],rowH:0.47,fontSize:9.5});
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
  ["1 · Sep–Oct","Supervision at scale — DONE","all 51 shards trained: 8,306 turns, 24 seeded runs, 0.1718 → 0.2065","0.1718 — cleared, p=4.1e-09"],
  ["2 · Oct–Nov","Features, not heads","exp08/11 show the head is not the insertion bottleneck — test text-conditioned features","a random-basis control of equal size"],
  ["3 · Nov–Dec","Editing and evaluation","mask-conditioned inpainting; collateral change outside the region — Review 2","MagicBrush mask at +18.5% net"]];
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

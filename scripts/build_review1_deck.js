const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const W = 13.333, H = 7.5;

const PAPER="FFFFFF", INK="1A1A1A", MUTE="6E6E6E", FAINT="9A9A9A",
      RULE="DEDCD7", TINT="F6F5F2", ACC="8C3A1E";
const HEAD="Cambria", BODY="Calibri";
const M = 0.85, CW = W - 2*M;

pres.author = "Amritha S (23BEC1368); Yugeshwaran P (23BEC1404)";
pres.title  = "Instruct-Seg-Edit - Review 1";

let N = 0;
function slide(kicker, title, sub){
  const s = pres.addSlide(); s.background = {color:PAPER}; N++;
  if(kicker) s.addText(kicker.toUpperCase(),{x:M,y:0.46,w:9,h:0.24,fontFace:BODY,fontSize:10.5,bold:true,color:ACC,charSpacing:2.2,margin:0});
  if(title)  s.addText(title,{x:M,y:0.76,w:CW,h:0.66,fontFace:HEAD,fontSize:31,bold:true,color:INK,margin:0});
  if(sub)    s.addText(sub,{x:M,y:1.46,w:CW,h:0.38,fontFace:BODY,fontSize:14,color:MUTE,italics:true,margin:0});
  s.addShape(pres.ShapeType.rect,{x:M,y:H-0.62,w:CW,h:0.008,fill:{color:RULE}});
  s.addText("Instruct-Seg-Edit  ·  Foundations of Data Science  ·  Review 1",{x:M,y:H-0.52,w:8,h:0.28,fontFace:BODY,fontSize:9,color:FAINT,margin:0});
  s.addText(String(N),{x:W-M-0.6,y:H-0.52,w:0.6,h:0.28,align:"right",fontFace:BODY,fontSize:9,color:FAINT,margin:0});
  return s;
}
const tint=(s,x,y,w,h)=>s.addShape(pres.ShapeType.rect,{x,y,w,h,fill:{color:TINT}});
const lab=(s,x,y,w,t,o={})=>s.addText(t,{x,y,w,h:0.26,fontFace:BODY,fontSize:o.size||13,bold:true,color:o.color||INK,margin:0});
function tbl(s,rows,opts={}){
  const c=opts.centerCols||[], br=opts.boldRows||[];
  const shaped=rows.map((r,ri)=>r.map((cell,ci)=>{
    const isObj=cell!==null&&typeof cell==="object";
    const o=Object.assign({},isObj?cell.options:{});
    if(ri===0){o.bold=true;o.fill={color:TINT};}
    if(br.indexOf(ri)!==-1)o.bold=true;
    if(c.indexOf(ci)!==-1)o.align="center";
    return {text:isObj?cell.text:cell,options:o};
  }));
  const t=Object.assign({fontFace:BODY,fontSize:10.5,color:INK,valign:"middle",
    border:{type:"solid",pt:0.5,color:RULE},fill:{color:PAPER},margin:[4,7,4,7]},opts);
  delete t.centerCols; delete t.boldRows;
  s.addTable(shaped,t);
}

/* ===== 1 TITLE ===== */
{
  const s=pres.addSlide(); s.background={color:PAPER}; N++;
  s.addText("VELLORE INSTITUTE OF TECHNOLOGY, CHENNAI",{x:M,y:1.2,w:11,h:0.3,fontFace:BODY,fontSize:11.5,bold:true,color:MUTE,charSpacing:1.6,margin:0});
  s.addText("School of Electronics Engineering  ·  Foundations of Data Science",{x:M,y:1.51,w:11,h:0.3,fontFace:BODY,fontSize:11.5,color:FAINT,margin:0});
  s.addText("Instruct-Seg-Edit",{x:M,y:2.35,w:11.6,h:1.15,fontFace:HEAD,fontSize:58,bold:true,color:INK,margin:0});
  s.addText("Disentangling Localization from Editing in Instruction-Guided Image Manipulation",
    {x:M,y:3.5,w:9.8,h:0.7,fontFace:HEAD,fontSize:19,italics:true,color:MUTE,lineSpacing:26,margin:0});
  s.addShape(pres.ShapeType.rect,{x:M,y:4.66,w:CW,h:0.008,fill:{color:RULE}});
  s.addText("PROJECT REVIEW 1",{x:M,y:4.9,w:4,h:0.24,fontFace:BODY,fontSize:10,bold:true,color:ACC,charSpacing:2,margin:0});
  s.addText([
    {text:"Amritha S",options:{bold:true,color:INK}},
    {text:"     23BEC1368",options:{color:MUTE,breakLine:true}},
    {text:"Yugeshwaran P",options:{bold:true,color:INK}},
    {text:"     23BEC1404",options:{color:MUTE}}
  ],{x:M,y:5.28,w:5.4,h:0.8,fontFace:BODY,fontSize:15,lineSpacing:24,margin:0});
  s.addText([
    {text:"Guided by",options:{color:FAINT,fontSize:10.5,bold:true,charSpacing:1.4,breakLine:true}},
    {text:"Dr. Saranya M",options:{color:INK,fontSize:15,bold:true,breakLine:true}},
    {text:"Employee ID 54783  ·  SCOPE",options:{color:MUTE,fontSize:12}}
  ],{x:6.8,y:5.28,w:6,h:0.9,fontFace:BODY,lineSpacing:22,margin:0});
  s.addText("Academic Year 2025 - 2026",{x:M,y:6.62,w:6,h:0.3,fontFace:BODY,fontSize:10.5,color:FAINT,margin:0});
  s.addNotes(`SLIDE 1   |   0:00 - 0:20   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   0:00 - 0:10 - - -
"Good morning. We are Amritha and Yugeshwaran, and this is Instruct-Seg-Edit. In one line: today's AI image editors decide where to edit and what to edit inside a single network."

- - - YUGESHWARAN  (23BEC1404)   0:10 - 0:20 - - -
"And our project separates those two decisions. We will both speak on every slide - Amritha opens each one and I follow with what it means for the design."

DELIVERY - Do not read the title, they can see it. Open on the claim.`);
}

/* ===== 2 PROBLEM AND GOAL ===== */
{
  const s=slide("Problem and goal","Editors change what you never asked about",null);
  tint(s,M,1.95,CW,1.4);
  s.addText("Instruction-guided diffusion editors decide WHERE to edit implicitly, inside the same network that performs the edit - so edits leak, and the spatial decision cannot be seen, measured or corrected.",
    {x:M+0.34,y:2.1,w:CW-0.68,h:1.1,fontFace:HEAD,fontSize:15.5,color:INK,lineSpacing:22,margin:0});
  s.addText("RESEARCH QUESTION",{x:M,y:3.52,w:5,h:0.24,fontFace:BODY,fontSize:10,bold:true,color:ACC,charSpacing:1.8,margin:0});
  s.addText("How do we separate the decision of where to edit from the execution of what to edit, and let the user correct the first before the second runs?",
    {x:M,y:3.8,w:CW,h:0.6,fontFace:HEAD,fontSize:15,italics:true,color:INK,lineSpacing:22,margin:0});
  lab(s,M,4.62,6,"Goal - what we set out to build");
  const g=[["Derive the supervision","No public dataset pairs an instruction with the region it changes"],
           ["Predict the edit region","A model that maps image plus instruction to a mask, at interactive speed"],
           ["Confine the edit","Generation restricted to that mask, so nothing outside it can change"],
           ["Keep the user in the loop","The mask is shown and can be corrected before generation runs"]];
  let y=4.96;
  g.forEach((v,i)=>{
    s.addText(String(i+1),{x:M,y,w:0.3,h:0.26,fontFace:HEAD,fontSize:12,bold:true,color:ACC,margin:0});
    s.addText(v[0],{x:M+0.42,y,w:3.0,h:0.26,fontFace:BODY,fontSize:12,bold:true,color:INK,margin:0});
    s.addText(v[1],{x:M+3.6,y,w:8.0,h:0.26,fontFace:BODY,fontSize:11.5,color:MUTE,margin:0});
    y+=0.35;
  });
  s.addNotes(`SLIDE 2   |   0:20 - 1:05   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   0:20 - 0:45 - - -
"Here is the problem. Ask a diffusion editor to put a hat on someone and it often changes their shirt or the background too. That happens because localization is implicit - it lives inside the same network that generates, and nothing in the objective requires the change to stay where you asked. So the spatial decision cannot be seen, measured, or corrected. That gives us our research question, on screen."

- - - YUGESHWARAN  (23BEC1404)   0:45 - 1:05 - - -
"And from that, four goals. Derive the supervision, because no public dataset pairs an instruction with the region it changes. Predict the edit region at interactive speed. Confine generation to that region. And keep the user in the loop, so the mask can be corrected before anything is generated."

DELIVERY - Amritha owns the problem and the research question; Yugeshwaran reads the four goals. Pause after the question.`);
}
/* ===== 2 INSPIRATION ===== */
{
  const s=slide("Inspiration","Foundation models are fine-tuned, not used raw",
    "The idea we are carrying over from text into images.");
  tint(s,M,2.0,CW,1.5);
  s.addText("A foundation model like BERT is a sequence model - given words, it predicts the next probable word. On its own that is not very useful. It becomes useful when you fine-tune it and connect it to a task: sentiment analysis, tone, or a conversational model.",
    {x:M+0.34,y:2.18,w:CW-0.68,h:1.2,fontFace:HEAD,fontSize:16,color:INK,lineSpacing:23,margin:0});
  const cols=[
    ["In text","BERT predicts the next word. Fine-tune it and attach a head, and it does sentiment analysis, tone detection, or dialogue - a conversational model."],
    ["In images","The same move. Take a vision foundation model that already understands objects, and fine-tune it onto a new task instead of training from zero."],
    ["Why it matters","Fine-tuning an already-trained model is far more compute efficient than training one, and in several tasks it performs better."]
  ];
  let x=M;
  cols.forEach(c=>{ lab(s,x,3.85,3.6,c[0]);
    s.addText(c[1],{x,y:4.19,w:3.5,h:1.35,fontFace:BODY,fontSize:12.5,color:MUTE,lineSpacing:18,margin:0}); x+=4.02; });
  s.addText("This project is that proof of concept for vision: fine-tune a foundation detection model onto a segmentation task, and use it to make image editing cheaper and more precise.",
    {x:M,y:5.65,w:CW,h:0.6,fontFace:HEAD,fontSize:15,italics:true,color:ACC,lineSpacing:21,margin:0});
  s.addNotes(`SLIDE 3   |   1:05 - 1:45   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   1:05 - 1:27 - - -
"Start with something familiar. A foundation model like BERT is a sequence model - given some words it predicts the next probable word. On its own that is not very useful. It becomes useful when you fine-tune it and connect it to a task: sentiment analysis, understanding tone, or a conversational model, which is what a chatbot is."

- - - YUGESHWARAN  (23BEC1404)   1:27 - 1:45 - - -
"We are doing the same move, but for images. Take a vision foundation model that already understands objects and fine-tune it onto a new task instead of training from scratch. That is the proof of concept: fine-tuning an already-trained model is far more compute efficient than training one, and in several tasks it actually performs better."

DELIVERY - Anchor in the text analogy first - the panel will follow images much faster once BERT has framed it.`);
}

/* ===== 4 LITERATURE SURVEY ===== */
{
  const s=slide("Literature survey","The base paper, its lineage, and the gap",
    "What we build on, and what none of it does.");
  const rows=[
    ["Work","What it contributes","Role in this project"],
    [{text:"FastSAM\nZhao et al., 2023   BASE PAPER",options:{bold:true}},
     "Rebuilds Segment Anything on a YOLOv8-seg backbone, trained on 2% of SA-1B, reaching comparable quality at roughly 50x the speed. Accepts text prompts through CLIP",
     "The foundation model we fine-tune. Our checkpoint is a fine-tune of FastSAM"],
    ["SAM\nKirillov et al., ICCV 2023",
     "Promptable segmentation pre-trained on 1 billion masks, with strong zero-shot transfer",
     "What FastSAM accelerates. Its native prompts are points and boxes, not language"],
    ["YOLOv8-seg\nJocher et al., 2023",
     "C2f backbone, PAN neck, anchor-free head with DFL, and a YOLACT-style prototype mask head",
     "The architecture underneath FastSAM, itself pre-trained on MS COCO"],
    ["Inpainting family\nSD, ControlNet, FLUX.1 Fill",
     "Mask-conditioned generation - the edit is confined to a region supplied as input",
     "Stage 2. It consumes the mask, but cannot derive one itself"]
  ];
  tbl(s,rows,{x:M,y:2.05,w:CW,colW:[2.35,5.1,4.18],rowH:0.72,fontSize:10});
  tint(s,M,5.45,CW,1.05);
  s.addText("THE GAP",{x:M+0.32,y:5.62,w:3,h:0.24,fontFace:BODY,fontSize:10,bold:true,color:ACC,charSpacing:1.8,margin:0});
  s.addText("Segmentation models localise what a prompt names; inpainting models edit a region they are handed. Nothing in this lineage links a natural-language instruction to the region an edit would actually change - which is the connection we propose to build.",
    {x:M+0.32,y:5.88,w:CW-0.64,h:0.55,fontFace:BODY,fontSize:11.5,color:INK,lineSpacing:16,margin:0});
  s.addNotes(`SLIDE 4   |   1:45 - 2:35   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   1:45 - 2:10 - - -
"Our base paper is FastSAM, from 2023. Segment Anything gave the field promptable segmentation trained on a billion masks, but it is slow. FastSAM rebuilds that same capability on a YOLOv8-seg backbone, trained on just two percent of SA-1B, and reaches comparable quality at roughly fifty times the speed - and it accepts text prompts through CLIP. That is the foundation model we fine-tune; our checkpoint is a fine-tune of FastSAM."

- - - YUGESHWARAN  (23BEC1404)   2:10 - 2:35 - - -
"Underneath FastSAM is YOLOv8-seg, which is itself pre-trained on MS COCO - so the lineage runs COCO, then SA-1B, then our own fine-tune. On the editing side we use the inpainting family: Stable Diffusion, ControlNet, FLUX.1 Fill. And that is where the gap is. Segmentation models localise what a prompt names, and inpainting models edit a region they are handed. Nothing in this lineage links an instruction to the region an edit would actually change."

DELIVERY - name FastSAM clearly as the base paper. If asked why not SAM directly: FastSAM is the one that is fast enough and takes text.`);
}

/* ===== 5 THE TASK ===== */
{
  const s=slide("The task","Why segmentation, and why inpainting",
    "Two shifts: from a box to an outline, and from denoising everything to denoising a region.");
  tint(s,M,2.0,5.62,2.45);
  lab(s,M+0.3,2.22,5,"From bounding box to outline",{size:12});
  s.addText("YOLO is an object detection model - ask for the car and you get a rectangle, not the car. The backbone carries a sense of shape that a box discards. With several vehicles in frame we must mark exactly the one named, pixel by pixel.",
    {x:M+0.3,y:2.58,w:5,h:1.25,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
  s.addText("we use a detection model to do a segmentation task",{x:M+0.3,y:3.9,w:5.05,h:0.45,fontFace:HEAD,fontSize:13.5,bold:true,color:ACC,lineSpacing:19,margin:0});
  tint(s,M+6.0,2.0,5.62,2.45);
  lab(s,M+6.3,2.22,5,"From denoising all, to denoising a region",{size:12});
  s.addText("A diffusion model generates by denoising - it starts from noise and removes it step by step. Denoise the whole frame and regions nobody asked to change are regenerated too, losing fine detail and artefacts.",
    {x:M+6.3,y:2.58,w:5,h:1.25,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
  s.addText("inpainting denoises only the cutout",{x:M+6.3,y:3.9,w:5.05,h:0.45,fontFace:HEAD,fontSize:13.5,bold:true,color:ACC,lineSpacing:19,margin:0});
  s.addText("Segmentation builds on top of detection - the model still has to find the object before it can fill in its outline - so we keep the detection backbone and add to it rather than replacing it.",
    {x:M,y:4.65,w:CW,h:0.5,fontFace:BODY,fontSize:12.5,color:MUTE,lineSpacing:18,margin:0});
  tint(s,M,5.25,CW,1.32);
  s.addText("Two problems solved at once",{x:M+0.32,y:5.48,w:6,h:0.28,fontFace:BODY,fontSize:12.5,bold:true,color:INK,margin:0});
  s.addText("Compute - we fine-tune a model that is already trained rather than training one, and then denoise only the masked region instead of the whole frame. Quality - the inpainting stage finally gets the precise segment it needs, so the edit is small, localized, and does no damage to pixels outside it.",
    {x:M+0.32,y:5.74,w:CW-0.64,h:0.8,fontFace:BODY,fontSize:11,color:MUTE,lineSpacing:15,margin:0});
  s.addNotes(`SLIDE 5   |   2:35 - 3:20   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   2:35 - 2:57 - - -
"Two shifts follow. The first is from a bounding box to an outline. YOLO gives you a rectangle around the car, not the car - the backbone carries a sense of shape that a box discards. With several vehicles in frame we must mark exactly the one named, pixel by pixel. So we use an object detection model to perform a segmentation task."

- - - YUGESHWARAN  (23BEC1404)   2:57 - 3:20 - - -
"The second is about generation. A diffusion model generates by denoising - it starts from noise and removes it step by step. Denoise the whole frame and regions nobody asked to change get regenerated, losing fine detail. Inpainting denoises only the cutout. So we solve two problems at once: compute, because we fine-tune rather than train and denoise only the region; and quality, because the inpainting stage finally gets the precise segment it needs."

DELIVERY - One shift each. The bottom block is the payoff - say it as the conclusion of both halves.`);
}
/* ===== 5 ARCHITECTURE ===== */
{
  const s=slide("Architecture","What we add on top of the detection backbone",
    "The base network stays; the new capability is bolted on.");
  const rows=[
    ["Component","What it is","Why"],
    ["Detection backbone","The YOLO network, 21 layers, already trained to find objects","Foundation model. We fine-tune rather than train from zero"],
    ["Segmentation head","Additional layers we add, which learn to produce the mask","Turns a detector into a segmenter without discarding detection"],
    ["CLIP text encoder","Encodes the user's instruction and fuses it with the image features","So the mask is chosen by language, not by class label"],
    ["Total","71.8 million parameters","Small enough to stay interactive"]
  ];
  tbl(s,rows,{x:M,y:2.05,w:CW,colW:[2.5,4.6,4.53],rowH:0.6,fontSize:11,boldRows:[4]});
  tint(s,M,5.35,CW,1.1);
  s.addText("Balanced multi-task weighting - the model must still detect before it can segment, so the segmentation task must not overshoot the detection task. Because the segmentation head is the part we added, we control how much of the training effort goes into it: if segmentation improves while detection degrades, the weighting is shifted back.",
    {x:M+0.32,y:5.55,w:CW-0.64,h:0.75,fontFace:BODY,fontSize:12,color:INK,lineSpacing:17,margin:0});
  s.addNotes(`SLIDE 6   |   3:20 - 4:05   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   3:20 - 3:42 - - -
"Architecturally we keep the base network and add to it. The detection backbone is the YOLO network, twenty-one layers, already trained to find objects - that is our foundation model. On top of it we add a segmentation head, extra layers that learn to produce the mask."

- - - YUGESHWARAN  (23BEC1404)   3:42 - 4:05 - - -
"And we add a CLIP text encoder, which encodes the user's instruction and fuses it with the image features, so the region is chosen by language rather than by a fixed class label. That comes to 71.8 million parameters. The balance matters - the model must still detect before it can segment, so the segmentation task must not overshoot detection. Since the head is the part we added, we control how much training effort goes into it."

DELIVERY - Balanced multi-task weighting is a real design decision - do not skip it, it shows judgement.`);
}

/* ===== 7 DATA AND LOSSES ===== */
{
  const s=slide("Data and training objective","How the data is represented, and the four losses",null);
  lab(s,M,2.0,6,"An image is a 3D matrix, a mask is a 2D one");
  s.addText("Think of a frame as a three-layer cake - one layer each for red, green and blue. Every element is a pixel's colour intensity from 0 to 255, so 256 values. Deep learning does not use that raw range because it costs too much memory, so values are scaled to 0 - 1. The mask is a single 2D matrix of the same width and height, and each element is binary: in the region, or not.",
    {x:M,y:2.34,w:CW,h:0.8,fontFace:BODY,fontSize:12,color:MUTE,lineSpacing:17,margin:0});
  lab(s,M,3.3,6,"Four losses, because it is a multi-task model");
  const rows=[
    ["Loss","Used for","What it measures"],
    ["CIoU","Bounding box regression","Overlap between predicted and ground-truth box. The score is how much they intersect; the loss is the inverse - the part that does not intersect is the error"],
    ["DFL  Distribution Focal Loss","Sharpening box coordinates","Instead of guessing one coordinate, the model predicts a distribution over where the edge can lie, and is taught the range it should fall in"],
    ["BCE  Binary Cross Entropy","Objectness and classification","A yes / no signal - if the answer is yes and the model says no it is penalised.  y·log(y') + (1-y)·log(1-y')"],
    ["Mask loss","The segmentation head","How well the predicted mask matches the true mask, inside the detected region"]
  ];
  tbl(s,rows,{x:M,y:3.64,w:CW,colW:[2.5,2.6,6.53],rowH:0.52,fontSize:10});
  tint(s,M,5.98,CW,0.72);
  s.addText("Balanced multi-task weighting - the model must detect before it can segment, so too much weight on the mask and detection degrades, leaving nothing to segment.",
    {x:M+0.32,y:6.12,w:CW-0.64,h:0.5,fontFace:BODY,fontSize:10.5,color:INK,lineSpacing:14,margin:0});
  s.addNotes(`SLIDE 7   |   4:05 - 4:50   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   4:05 - 4:25 - - -
"A word on representation, because the losses only make sense afterwards. An image is a three-layer cake - red, green and blue - and every element is a pixel intensity from 0 to 255. Deep learning does not use that raw range because it costs too much memory, so we scale to 0 to 1. The mask is a single 2D matrix of the same size, and each element is binary: in the region or not."

- - - YUGESHWARAN  (23BEC1404)   4:25 - 4:50 - - -
"It trains against four losses. CIoU for the bounding box - the score is how much predicted and true boxes overlap, and the loss is the inverse. Distribution Focal Loss, where instead of guessing one coordinate the model predicts a distribution over where the edge can lie. Binary cross entropy for objectness and classification, a yes-no signal. And a mask loss for the segmentation head. They are summed with a weighting - too much on the mask and detection degrades, leaving nothing to segment."

DELIVERY - Name each loss and what it measures. Do not read the formula aloud.`);
}
/* ===== APPROACH ===== */
{
  const s=slide("Proposed approach","The pipeline we propose",
    "Two models with a human step between them, and the behaviour we are aiming for.");
  const b=[
    ["STAGE 1","Localization model","image + instruction","binary edit mask","FastSAM, a YOLOv8-seg architecture of 71.8M parameters, prompted with the instruction text."],
    ["INTERVENTION","Human refinement","predicted mask","corrected mask","A brush-editable mask canvas. Wrong object? Fix it locally, not by rewording."],
    ["STAGE 2","Mask-conditioned edit","image + instruction + mask","edited image","Generation confined to the mask, so nothing outside it can change."]
  ];
  let x=M;
  b.forEach((v,i)=>{
    const w=3.62;
    tint(s,x,2.0,w,2.35);
    s.addText(v[0],{x:x+0.28,y:2.25,w:w-0.5,h:0.24,fontFace:BODY,fontSize:9.5,bold:true,color:ACC,charSpacing:1.8,margin:0});
    s.addText(v[1],{x:x+0.28,y:2.52,w:w-0.5,h:0.55,fontFace:HEAD,fontSize:15,bold:true,color:INK,lineSpacing:22,margin:0});
    s.addText([{text:"in    ",options:{color:FAINT}},{text:v[2],options:{color:INK,breakLine:true}},
               {text:"out  ",options:{color:FAINT}},{text:v[3],options:{color:INK}}],
      {x:x+0.28,y:3.12,w:w-0.5,h:0.55,fontFace:BODY,fontSize:10.5,lineSpacing:15,margin:0});
    s.addText(v[4],{x:x+0.28,y:3.72,w:w-0.5,h:0.6,fontFace:BODY,fontSize:10.5,color:MUTE,lineSpacing:14,margin:0});
    if(i<2) s.addText("→",{x:x+w+0.04,y:3.05,w:0.36,h:0.3,align:"center",fontFace:BODY,fontSize:16,color:FAINT,margin:0});
    x+=w+0.44;
  });
  {
    const iw=3.62, ih=1.92, iy=4.55;
    const panels=[["img/a_original.png","Input image, plus the instruction"],
                  ["img/b_mask.png","The mask Stage 1 must produce - the table alone"],
                  ["img/c_edited.png","Target - the table changes, the plant and jars do not"]];
    let ix=M;
    panels.forEach(pn=>{
      s.addImage({path:pn[0], x:ix, y:iy, w:iw, h:ih});
      s.addText(pn[1],{x:ix,y:iy+ih+0.05,w:iw,h:0.3,fontFace:BODY,fontSize:9.5,color:MUTE,lineSpacing:12,margin:0});
      ix+=iw+0.44;
    });
  }
  s.addNotes(`SLIDE 8   |   4:50 - 5:40   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   4:50 - 5:15 - - -
"This is the pipeline we have designed. Stage one is a localization model - image and instruction in, a binary edit mask out, supervised on the region an edit changes. The mask is then shown to the user and can be corrected. Stage two generates, confined to that mask."

- - - YUGESHWARAN  (23BEC1404)   5:15 - 5:40 - - -
"The strip along the bottom illustrates the behaviour we are aiming for. On the left, the input image with an instruction to recolour the table. In the middle, the mask Stage 1 has to produce - the table alone, following its actual shape rather than a bounding box. On the right, the outcome we want: the table changes, and the plant and the two jars standing on it are untouched. Those objects are the test - a model that edits the whole region would have changed them too."

DELIVERY - present these as the target, not as a finished result. Point at the background in the third panel - that is the property we are designing for.`);
}

/* ===== 9 DATASET AND PLAN ===== */
{
  const s=slide("Dataset and plan","What we train on, and what we plan to achieve",null);
  lab(s,M,2.0,6,"Training lineage - each stage inherits from the one above");
  const ds=[["Stage","Data","What it gives"],
            ["YOLOv8-seg pre-training","MS COCO","118,287 images, ~860,000 instances, 80 categories - general object understanding"],
            ["FastSAM training","2% of SA-1B","Class-agnostic segmentation of anything, at real-time speed"],
            ["Our fine-tune","SA-1B","A class-agnostic checkpoint specialised for the regions our editor needs"]];
  tbl(s,ds,{x:M,y:2.34,w:CW,colW:[2.6,1.9,7.13],rowH:0.42,fontSize:10});
  const f=[["Hardware","NVIDIA A100"],["Epochs","7"],["Training time","~50 hours"]];
  let x=M;
  f.forEach(v=>{ s.addText(v[1],{x,y:3.92,w:3.2,h:0.42,fontFace:HEAD,fontSize:18,bold:true,color:ACC,margin:0});
                 s.addText(v[0],{x,y:4.36,w:3.2,h:0.26,fontFace:BODY,fontSize:11,color:MUTE,margin:0}); x+=3.9; });
  s.addShape(pres.ShapeType.rect,{x:M,y:4.8,w:CW,h:0.008,fill:{color:RULE}});
  lab(s,M,5.0,6,"What we plan to achieve - three months");
  const plan=[["Month","Focus","Deliverable"],
    ["1  ·  Aug - Sep","Dataset","Instruction-to-region pairs completed, held-out evaluation split fixed"],
    ["2  ·  Sep - Oct","Localization","Segmentation head fine-tuned and evaluated - IoU, Dice, precision, recall"],
    ["3  ·  Oct - Nov","Editing","Inpainting quality measured against editing the whole frame - Review 2"]];
  tbl(s,plan,{x:M,y:5.34,w:CW,colW:[1.9,2.2,7.53],rowH:0.36,fontSize:10});
  s.addNotes(`SLIDE 9   |   5:40 - 6:25   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   5:40 - 6:02 - - -
"On data, it is worth showing the whole lineage rather than one dataset, because each stage inherits from the one above. YOLOv8-seg is pre-trained on MS COCO - 118,287 images, around 860,000 annotated instances across 80 categories - which is where the general object understanding comes from. FastSAM is then trained on two percent of SA-1B, which is what makes it class-agnostic. Our fine-tune sits on top of that."

- - - YUGESHWARAN  (23BEC1404)   6:02 - 6:25 - - -
"That fine-tune ran on an NVIDIA A100, seven epochs, about fifty hours. And here is the plan across three months. Month one, complete the instruction-to-region pairs and fix a held-out split. Month two, fine-tune and evaluate the localization with IoU, Dice, precision and recall. Month three, measure the inpainting quality against editing the whole frame - and that is what we bring to Review 2."

DELIVERY - the lineage answers the COCO-versus-SA-1B question before it is asked. Both are true, at different stages.`);
}
/* ===== 10 THANK YOU ===== */
{
  const s=pres.addSlide(); s.background={color:PAPER}; N++;
  s.addText("Thank you",{x:M,y:2.35,w:11.6,h:1.1,fontFace:HEAD,fontSize:54,bold:true,color:INK,margin:0});
  s.addText("We are happy to take questions.",{x:M,y:3.5,w:9,h:0.5,fontFace:HEAD,fontSize:19,italics:true,color:MUTE,margin:0});
  s.addShape(pres.ShapeType.rect,{x:M,y:4.5,w:CW,h:0.008,fill:{color:RULE}});
  s.addText([
    {text:"Amritha S",options:{bold:true,color:INK}},
    {text:"     23BEC1368",options:{color:MUTE,breakLine:true}},
    {text:"Yugeshwaran P",options:{bold:true,color:INK}},
    {text:"     23BEC1404",options:{color:MUTE}}
  ],{x:M,y:4.85,w:5.4,h:0.8,fontFace:BODY,fontSize:15,lineSpacing:24,margin:0});
  s.addText([
    {text:"Guided by",options:{color:FAINT,fontSize:10.5,bold:true,charSpacing:1.4,breakLine:true}},
    {text:"Dr. Saranya M",options:{color:INK,fontSize:15,bold:true,breakLine:true}},
    {text:"Employee ID 54783  ·  SCOPE",options:{color:MUTE,fontSize:12}}
  ],{x:6.8,y:4.85,w:6,h:0.9,fontFace:BODY,lineSpacing:22,margin:0});
  s.addText("Instruct-Seg-Edit  ·  Foundations of Data Science  ·  Review 1  ·  AY 2025-2026",
    {x:M,y:6.5,w:10,h:0.3,fontFace:BODY,fontSize:10.5,color:FAINT,margin:0});
  s.addNotes(`SLIDE 10   |   6:25 - 6:45   |   BOTH SPEAK

- - - AMRITHA  (23BEC1368)   6:25 - 6:36 - - -
"To close, the one sentence we would like you to take away: localization and editing are two different decisions, and making the first one explicit is what buys supervision, measurement, and user control."

- - - YUGESHWARAN  (23BEC1404)   6:36 - 6:45 - - -
"Nothing in the lineage we build on does that. Thank you for listening - we are happy to take your questions."

DELIVERY - close together, facing the panel, then stop talking and let them ask.

-------------------------------------------------------------------
IF ASKED WHAT THE BASE PAPER IS
  FastSAM (Zhao et al., 2023). It rebuilds Segment Anything on a
  YOLOv8-seg backbone, trained on 2% of SA-1B, at roughly 50x SAM's
  speed, and takes text prompts through CLIP. Our checkpoint is a
  fine-tune of it - 71.8M parameters, class-agnostic.
  Do NOT cite InstructPix2Pix: it appears nowhere in the codebase.

IF ASKED WHICH DATASET - all three are true, at different stages:
  MS COCO      pre-trains YOLOv8-seg (118,287 images, 80 classes)
  SA-1B (2%)   trains FastSAM, which is what makes it class-agnostic
  SA-1B        our own fine-tune (the checkpoint's arg is sa.yaml)

IF ASKED WHAT ALREADY RUNS - the pipeline works end to end on the
repository's "app" branch:
  - Stage 1: FastSAM prompted with instruction text
  - Refinement: React front end with a brush-editable mask canvas
  - Stage 2: inpainting, six backends - FLUX.1 Fill (int4), SD3 +
    ControlNet, SDXL, SD2, SD1.5 + ControlNet, and a hosted API
  - Serving: FastAPI /api/segment and /api/inpaint, plus cli.py

WHAT IS GENUINELY NOT DONE: the evaluation. No IoU, Dice, LPIPS or
FID numbers exist yet, and there is no comparison against editing the
whole frame. That is exactly what slide 9 commits to.
-------------------------------------------------------------------`);
}

pres.writeFile({fileName:"Instruct-Seg-Edit_Review1.pptx"}).then(f=>console.log("wrote",f));

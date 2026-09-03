const fs=require("fs");
const {Document,Packer,Paragraph,TextRun,AlignmentType,Table,TableRow,TableCell,WidthType,
       ShadingType,BorderStyle,PageBreak}=require("docx");
const D=JSON.parse(fs.readFileSync("speech.json","utf8"));
const SERIF="Cambria", SANS="Calibri", ACC="8C3A1E", MUTE="6E6E6E";
const CW=9746;

const P=(t,o={})=>new Paragraph({alignment:o.align||AlignmentType.LEFT,
  spacing:{after:o.after===undefined?100:o.after,line:o.line||276},
  children:[new TextRun({text:t,font:o.font||SANS,size:o.size||21,bold:o.bold,italics:o.i,color:o.color})]});

function speechRow(who,time,text){
  const isA = who==="AMRITHA";
  return new Table({
    columnWidths:[1750,7996], width:{size:CW,type:WidthType.DXA},
    borders:{top:{style:BorderStyle.NONE},bottom:{style:BorderStyle.NONE},
             left:{style:BorderStyle.NONE},right:{style:BorderStyle.NONE},
             insideHorizontal:{style:BorderStyle.NONE},insideVertical:{style:BorderStyle.NONE}},
    rows:[new TableRow({children:[
      new TableCell({width:{size:1750,type:WidthType.DXA},
        shading:{type:ShadingType.CLEAR,fill:isA?"F6F1EE":"F1F3F5",color:"auto"},
        margins:{top:100,bottom:100,left:120,right:100},
        children:[
          new Paragraph({spacing:{after:20},children:[new TextRun({text:isA?"AMRITHA":"YUGESHWARAN",
            font:SANS,size:16,bold:true,color:isA?ACC:"33566E"})]}),
          new Paragraph({spacing:{after:0},children:[new TextRun({text:isA?"23BEC1368":"23BEC1404",
            font:SANS,size:14,color:MUTE})]}),
          new Paragraph({spacing:{before:60,after:0},children:[new TextRun({text:time,font:SANS,size:15,color:MUTE})]})
        ]}),
      new TableCell({width:{size:7996,type:WidthType.DXA},
        margins:{top:100,bottom:100,left:180,right:120},
        children:[new Paragraph({spacing:{after:0,line:288},
          children:[new TextRun({text:'“'+text+'”',font:SERIF,size:22})]})]})
    ]})]});
}

const kids=[
  new Paragraph({spacing:{before:600,after:60},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:"INSTRUCT-SEG-EDIT",font:SERIF,size:40,bold:true})]}),
  new Paragraph({spacing:{after:40},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:"Review 1  ·  Presentation Script",font:SERIF,size:24,italics:true,color:MUTE})]}),
  new Paragraph({spacing:{after:400},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:"Amritha S  23BEC1368     ·     Yugeshwaran P  23BEC1404",font:SANS,size:20,color:MUTE})]}),
  P("How to use this","",{}),
];
kids.pop();
kids.push(P("How to use this",{bold:true,size:22,after:80}));
kids.push(P("Every slide has two speaking turns, one each, with a clock window. The quoted text is what you say; the line marked Delivery underneath is a stage direction and is not read aloud. Total scripted speech is about 7 minutes 20 seconds at a rehearsed pace, which leaves room inside a 10-minute slot for pauses and slide changes.",{size:20,color:MUTE,after:520}));

D.forEach(s=>{
  kids.push(new Paragraph({spacing:{before:420,after:30},
    children:[new TextRun({text:`SLIDE ${s.n}`,font:SANS,size:17,bold:true,color:ACC,characterSpacing:30}),
              new TextRun({text:`     ${s.window}`,font:SANS,size:17,color:MUTE})]}));
  kids.push(new Paragraph({spacing:{after:120},border:{bottom:{style:BorderStyle.SINGLE,size:4,color:"DEDCD7",space:6}},
    children:[new TextRun({text:(s.n===1?"Title slide":s.title),font:SERIF,size:26,bold:true})]}));
  s.segs.forEach(([who,tm,txt])=>{ kids.push(speechRow(who,tm,txt)); kids.push(P("",{after:60,size:8})); });
  if(s.delivery) kids.push(new Paragraph({spacing:{after:60},indent:{left:1750},
    children:[new TextRun({text:"Delivery — ",font:SANS,size:18,bold:true,color:MUTE}),
              new TextRun({text:s.delivery,font:SANS,size:18,italics:true,color:MUTE})]}));
});

const doc=new Document({creator:"Amritha S; Yugeshwaran P",title:"Instruct-Seg-Edit — Review 1 Script",
  sections:[{properties:{page:{margin:{top:1080,right:1080,bottom:1080,left:1080}}},children:kids}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("Instruct-Seg-Edit_Review1_Script.docx",b);
  console.log("wrote Instruct-Seg-Edit_Review1_Script.docx");});

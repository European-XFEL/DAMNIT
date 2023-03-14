let divEl  = document.getElementById("mainDiv");
let startEl = document.getElementById("start");
let endEl = document.getElementById("end");
let autoScroll = true;

const stylesheet = document.styleSheets[0];
const logTypes = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

window.history.scrollRestoration = 'manual'

document.addEventListener('DOMContentLoaded', function() {
  window.onload = function() {
    let channel = new QWebChannel(qt.webChannelTransport, function(channel) {
      window.log_channel_obj = channel.objects.log_channel_obj;
    });
  }
});


function addLine(logLine, top = false) {
  let logType = "ERROR"
  const tag = document.createElement("pre");
  const br = document.createElement("br");
  const logLineText = document.createTextNode(logLine);

  for (let i in logTypes){
    if (logLine.includes(logTypes[i])){
      logType = logTypes[i]
    }
  }
 
  tag.appendChild(logLineText);
  tag.appendChild(br);
  tag.classList.add(logType);
  tag.classList.add("log");

  if (top) {
    if(startEl.nextElementSibling != null) {
      let pastEl = getPastDate();
      let pastDate = pastEl.textContent.slice(0,10)
      let pastDay = NaN
      if (pastDate != null) {
        pastDay = pastDate.slice(-2); 
      }
      let currentDate = tag.textContent.slice(0,10);  
      let currentDay = parseInt(currentDate.slice(-2).trim());
      if(pastDay != currentDay && !isNaN(currentDay) && !isNaN(pastDay)){
        createDivider(currentDate,pastDate, pastEl)
      }
    }
    divEl.insertBefore(tag,startEl.nextSibling);
    window.scrollTo(0, 0);
  } else {
    divEl.appendChild(tag);
    if (logType === "ERROR") {
      log_channel_obj.send_error_signal();
    }
    if (autoScroll) {
      window.scrollTo(0, document.body.scrollHeight);
    };
  };
};

function getPastDate(){
  let pastEl = startEl.nextElementSibling;
  let noText = pastEl.textContent == undefined;
  let tooManyLines = false
  let noDate = true
  let pastDay = ''
  let i = 0
  const N = 50
  if (!noText){
    pastDay = pastEl.textContent.slice(8,10);
    noDate = isNaN(parseInt(pastDay));
  } else {
    noDate = true
  }
  while (noDate || tooManyLines){
    pastEl = pastEl.nextElementSibling;
    if (pastEl == null){
      return null
    }
    noText = pastEl.textContent == null;
    if (!noText){
      pastDay = pastEl.textContent.slice(8,10);
      noDate = isNaN(parseInt(pastDay));
    } else {
      noDate = true
    }
    i+=1
    tooManyLines = i>N
  }
  return pastEl;
}

function changeDisplay(logType, hide = false) {
  let elementRules;
  for (let i = 0; i< stylesheet.cssRules.length; i++) {
    console.log(stylesheet.cssRules[i].selectorText)
    if(stylesheet.cssRules[i].selectorText == '.'+logType) {
      elementRules = stylesheet.cssRules[i];
    }
  }

  if (hide) {
    elementRules.style.setProperty("display", "none");
  } else {
    elementRules.style.setProperty("display", "inline");
  }
}

function createDivider(pastDay = '',futureDay = '', El) {
  let divDivider = document.createElement("div");
  let divShadowArea =  document.createElement("div");
  let hrLineOne = document.createElement("hr");
  let hrLineTwo = document.createElement("hr");
  let textArea = document.createElement("span");
  let textContent = document.createTextNode(
    '↓ ' + futureDay + ' | ' + pastDay  + ' ↑'
  )

  divDivider.classList.add("page-divider");
  divShadowArea.classList.add("shaded-area");
  hrLineOne.classList.add("divider-line");
  hrLineTwo.classList.add("divider-line");
  textArea.classList.add("text-divider");

  divDivider.appendChild(hrLineOne);
  textArea.appendChild(textContent);
  divShadowArea.appendChild(textArea);
  divDivider.appendChild(divShadowArea);
  divDivider.appendChild(hrLineTwo);

  divEl.insertBefore(divDivider,El);
}

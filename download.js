const fs = require('fs')  
const path = require('path')  
const axios = require('axios')



//Server payloads
var payloads = [  
{
  id: "1588850712345",
  url: 'https://myfile.com/download/bin/e1.exe'
}, 

{
  id: "1588850776449",
  url: 'https://myfile.com/download/bin/e231.exe'
}]
for(let payload of payloads) {
  setTimeout(() => {}, 3000)
  console.log(payload.id)
}

// for (let [index] = 0; index < array.length; index++) {
//   const element = array[index];
  
// }

// payloads.forEach(payload => {
  
//   setTimeout(() => {
//     console.log(payload.id)
//   }, 3000);
// });

async function downloadAndExec () {  
  const url = 'https://unsplash.com/photos/AaEQmoufHLk/download?force=true'
  const dpath = path.resolve(__dirname, '')
  const writer = fs.createWriteStream(dpath)

  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream'
  })

  response.data.pipe(writer)

  return new Promise((resolve, reject) => {
    writer.on('finish', resolve)
    writer.on('error', reject)
  })
}

//downloadImage()
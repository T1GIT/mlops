import express from "express";
import axios from "axios";
import {engine} from 'express-handlebars'
import bodyParser from "body-parser";
import Handlebars from 'handlebars'

const PORT = 3000
const API_URL = `http://${process.env['API_URL'] ?? '127.0.0.1:8000'}`

const app = express()

app.use(bodyParser.json())

app.engine('handlebars', engine())
app.set('view engine', 'handlebars')
app.set('views', './views')


async function main() {
  const {data: meta} = await axios.get(`${API_URL}/meta`)
  const metaJson = new Handlebars.SafeString(JSON.stringify(meta))

  app.get('/', (req, res) => {
    res.render('index', { meta, metaJson })
  })

  app.post('/predict', async (req, res) => {
    const {data} = await axios.post(`${API_URL}/predict`, req.body)
    res.send(data)
  })

  app.post('/feedback', async (req, res) => {
    const {data} = await axios.post(`${API_URL}/feedback`, req.body)
    res.send(data)
  })

  app.listen(PORT, () => {
    console.log(`App listening on port ${PORT}`)
  })
}

void main();


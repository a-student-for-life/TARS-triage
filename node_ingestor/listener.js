/**
 * AIS WebSocket stream listener — forwards ship position reports to the Python backend.
 */
import "dotenv/config";
import WebSocket from "ws";
import axios from "axios";

// Configuration — set these in node_ingestor/.env (never commit that file)
const API_KEY = process.env.AIS_API_KEY || "";
const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:8000/ingest";
const AIS_WS_URL = "wss://stream.aisstream.io/v0/stream";

// aisstream.io sends e.g. "2026-01-01 18:19:09.123456789 +0000 UTC".
const toIsoUtc = (timeUtc) => {
    if (!timeUtc) return null;
    const m = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(\.\d+)?/.exec(timeUtc);
    if (!m) return null;
    const fraction = m[3] ? m[3].slice(0, 7) : "";
    return `${m[1]}T${m[2]}${fraction}Z`;
};

const connectToStream = () => {

    // create a websocket instance 
    const ws = new WebSocket(AIS_WS_URL);

    // create event listeners on the websocket instance

    // create event of opening ie connecting to the url and 
    // subscribing with credentials
    ws.on('open', () => {
        console.log("Connecting to the stream!!");
        const subscription = {
            APIKey : API_KEY,
            BoundingBoxes : [[[-90,-180], [90,180]]],
            FilterMessageTypes : ["PositionReport"]
        }
        ws.send(JSON.stringify(subscription));
    });

    // create event of getting message after opening
    ws.on('message', async (data) => {
        try{
            const msg = JSON.parse(data);
            if(msg.MessageType === "PositionReport"){
                const pos = msg.Message.PositionReport
                // Send raw values; the backend drops AIS "not available" codes
                // (SOG 102.3, lat 91, lon 181, COG 360) instead of treating them as real.
                const payload = {
                    mmsi : pos.UserID,
                    lat : pos.Latitude,
                    lon : pos.Longitude,
                    speed : pos.Sog,
                    course : pos.Cog,
                    timestamp : toIsoUtc(msg.MetaData && msg.MetaData.time_utc)
                };
                console.log(`ship ${payload.mmsi} received Sending to python`);
                await axios.post(PYTHON_BACKEND,payload);
            }
        }catch(error){
            console.error("Error forwarding it to Python:", error.message);
        }
    });

    //create event of errors that may happen
    ws.on('error', (err) => {
        console.log('error:', err.message);
    });

    //create event of closing the connection and recalling the connectToStream function
    ws.on('close', () => {
        console.log('Connection Timed out! Restarting in 5s');
        setTimeout(connectToStream, 5000);
    });
};

connectToStream();

import { useEffect, useState } from 'react'
import axios from 'axios'
import Map from './Map'
import './App.css'

const ARCHIVED_STATUSES = new Set(['SAFE', 'COMPLETED', 'RESOLVED', 'IGNORED'])

function App() {
  const [alerts, setAlerts] = useState([])
  const [notificationMsg, setNotificationMsg] = useState('')
  const [loadError, setLoadError] = useState('')

  const showNotification = (message) => {
    if (!message) return
    setNotificationMsg(message)
    setTimeout(() => setNotificationMsg(''), 4000)
  }

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'}/alerts`)
      setAlerts(response.data)
      setLoadError('')
    } catch (error) {
      console.error('Fetch error:', error)
      setLoadError('Backend unavailable. Unable to load active alerts.')
    }
  }

  const handleAlertAction = (updatedAlert, message) => {
    if (updatedAlert && updatedAlert.ship_id) {
      setAlerts((currentAlerts) => {
        if (ARCHIVED_STATUSES.has(updatedAlert.status)) {
          return currentAlerts.filter((alert) => alert.id !== updatedAlert.id)
        }

        const existingAlert = currentAlerts.find((alert) => alert.id === updatedAlert.id)
        if (!existingAlert) {
          return [updatedAlert, ...currentAlerts]
        }

        return currentAlerts.map((alert) => (
          alert.id === updatedAlert.id ? updatedAlert : alert
        ))
      })
    }

    showNotification(message)
  }

  useEffect(() => {
    fetchAlerts()
    const interval = setInterval(fetchAlerts, 3000)
    return () => clearInterval(interval)
  }, [])

  const locateSpill = () => {
    const spill = alerts.find((alert) => alert.final_result === 'OIL_DETECTED')
    if (spill) {
      window.dispatchEvent(new CustomEvent('mapFly', { detail: [spill.lat, spill.lon] }))
    } else {
      alert('No active oil spills detected.')
    }
  }

  return (
    <div className="main-wrapper">
      <div className="dashboard-overlay">
        <h2 style={{ margin: '0 0 10px 0' }}>Satellite Watchdog</h2>

        <div className="stats">
          <p>Active Threats: <b>{alerts.length}</b></p>
          <p>Spills Detected: <b style={{ color: 'red' }}>{alerts.filter((alert) => alert.final_result === 'OIL_DETECTED').length}</b></p>
        </div>

        <div className="legend">
          <p><span className="legend-dot red-dot"></span> Anomalous vessel</p>
          <p><span className="legend-dot pulse-dot"></span> Oil spill confirmed</p>
        </div>

        {notificationMsg && <div className="dispatch-msg">{notificationMsg}</div>}
        {loadError && <div className="dispatch-msg error-msg">{loadError}</div>}

        <button className="teleport-btn" onClick={locateSpill}>
          Locate Spill
        </button>
      </div>

      <Map alerts={alerts} onAlertAction={handleAlertAction} />
    </div>
  )
}

export default App

import React, { useEffect } from 'react';
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import axios from 'axios';

const ARCHIVED_STATUSES = new Set(['RESOLVED', 'IGNORED', 'SAFE', 'COMPLETED']);

const buildPinIcon = (className) => L.divIcon({
    className: 'custom-marker-wrapper',
    html: `<div class="custom-marker-pin ${className}"></div>`,
    iconSize: [24, 36],
    iconAnchor: [12, 36],
    popupAnchor: [0, -30]
});

const redIcon = buildPinIcon('marker-anomalous');
const greenIcon = buildPinIcon('marker-safe');
const imageFallbackDataUrl = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="300" height="150" viewBox="0 0 300 150">
  <rect width="300" height="150" fill="#f5f7fa" />
  <rect x="12" y="12" width="276" height="126" rx="10" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2" />
  <text x="150" y="80" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#475569">
    Satellite image unavailable
  </text>
</svg>
`)}`;

const pulsingIcon = L.divIcon({
    className: 'pulsing-marker',
    html: '<div class="ring"></div><div class="dot"></div>',
    iconSize: [20, 20],
    iconAnchor: [10, 10]
});

const MapController = () => {
    const map = useMap();
    useEffect(() => {
        const handleFly = (event) => {
            map.flyTo(event.detail, 12, { duration: 2.5 });
        };
        window.addEventListener('mapFly', handleFly);
        return () => window.removeEventListener('mapFly', handleFly);
    }, [map]);
    return null;
};

const formatTimestamp = (value) => {
    if (!value) return 'N/A';
    const normalizedValue = (
        typeof value === 'string' && !/[zZ]|[+-]\d{2}:\d{2}$/.test(value)
            ? `${value}Z`
            : value
    );
    const parsedDate = new Date(normalizedValue);
    if (Number.isNaN(parsedDate.getTime())) return 'N/A';
    return parsedDate.toLocaleString(undefined, { timeZone: 'UTC', timeZoneName: 'short' });
};

const getAlignmentClassName = (alignmentStatus) => {
    if (alignmentStatus === 'ALIGNED') return 'status-badge aligned';
    if (alignmentStatus === 'WEAK_ALIGNMENT') return 'status-badge weak';
    if (alignmentStatus === 'MISALIGNED') return 'status-badge misaligned';
    return 'status-badge neutral';
};

const getResultLabel = (alert) => {
    if (alert.final_result) return alert.final_result;
    if (alert.status === 'SEARCHING_IMG' || alert.image_status === 'PENDING') {
        return 'Searching satellite imagery';
    }
    if (alert.status === 'ANALYZING') {
        return 'Analyzing SAR image';
    }
    if (alert.image_status === 'FAILED') {
        return 'Satellite image download failed';
    }
    if (alert.image_status === 'UNAVAILABLE') {
        return 'No relevant SAR image available';
    }
    if (alert.image_alignment_status === 'MISALIGNED') {
        return 'Image misaligned with anomaly';
    }
    return 'Pending review';
};

const renderImagePanel = (alert) => {
    if (alert.image_url) {
        return (
            <>
                <img
                    src={alert.image_url}
                    alt="Satellite View"
                    className="satellite-img"
                    onError={(event) => {
                        event.currentTarget.src = imageFallbackDataUrl;
                    }}
                />
                {alert.image_alignment_status === 'WEAK_ALIGNMENT' && (
                    <p className="warning-note weak-note">
                        Weak temporal alignment. Used for inference with caution.
                    </p>
                )}
                {alert.image_alignment_status === 'MISALIGNED' && (
                    <p className="warning-note misaligned-note">
                        Misaligned image. Displayed for review only and not used for confirmation.
                    </p>
                )}
            </>
        );
    }

    if (alert.image_status === 'PENDING' || alert.status === 'SEARCHING_IMG' || alert.status === 'ANALYZING') {
        return <div className="image-placeholder">Loading satellite image...</div>;
    }

    return <div className="image-placeholder">No relevant satellite image available</div>;
};

const Map = ({ alerts, onAlertAction }) => {
    const position = [20, 0];

    const handleUpdateStatus = async (id, newStatus) => {
        try {
            const response = await axios.put(`${import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'}/alert/${id}/status`, { status: newStatus });
            const updatedAlert = response.data;

            let message = '';
            if (newStatus === 'DISPATCHED') {
                message = updatedAlert.dispatched_at
                    ? `Dispatch confirmed at ${formatTimestamp(updatedAlert.dispatched_at)}`
                    : 'Alert dispatched.';
            } else if (newStatus === 'IGNORED' || newStatus === 'RESOLVED') {
                message = 'Alert archived.';
            }

            onAlertAction(updatedAlert, message);
        } catch (error) {
            console.error('Update failed', error);
            onAlertAction(null, 'Action failed. Please retry.');
        }
    };

    return (
        <MapContainer center={position} zoom={2} style={{ height: '100vh', width: '100%' }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <MapController />

            <MarkerClusterGroup chunkedLoading>
                {alerts.map((alert) => {
                    const isCritical = alert.final_result === 'OIL_DETECTED';
                    const isSafe = alert.final_result === 'CLEAN_SEA';

                    let finalIcon = redIcon;
                    if (isCritical) finalIcon = pulsingIcon;
                    else if (isSafe) finalIcon = greenIcon;

                    return (
                        <Marker
                            key={`${alert.id}-${alert.status}-${alert.image_status || 'NA'}`}
                            position={[alert.lat, alert.lon]}
                            icon={finalIcon}
                        >
                            <Popup>
                                <div className="popup-content">
                                    <h3>MMSI: {alert.ship_id}</h3>

                                    <p>
                                        <b>Result:</b>{' '}
                                        <span style={{
                                            color: isCritical ? 'red' : (isSafe ? 'green' : 'orange'),
                                            fontWeight: 'bold'
                                        }}>
                                            {getResultLabel(alert)}
                                        </span>
                                    </p>

                                    <p><b>Status:</b> {alert.status}</p>
                                    <p><b>Reason:</b> {alert.anomaly_reason || alert.anomaly_type || 'N/A'}</p>
                                    <p><b>Confidence:</b> {alert.prediction_confidence != null ? `${alert.prediction_confidence.toFixed(2)}%` : 'N/A'}</p>

                                    {renderImagePanel(alert)}

                                    <div className="metadata-grid">
                                        <div className="metadata-row">
                                            <span>Anomaly time</span>
                                            <span>{formatTimestamp(alert.anomaly_timestamp)}</span>
                                        </div>
                                        <div className="metadata-row">
                                            <span>Image time</span>
                                            <span>{formatTimestamp(alert.image_timestamp)}</span>
                                        </div>
                                        <div className="metadata-row">
                                            <span>Image status</span>
                                            <span>{alert.image_status || 'N/A'}</span>
                                        </div>
                                        <div className="metadata-row">
                                            <span>Alignment</span>
                                            <span className={getAlignmentClassName(alert.image_alignment_status)}>
                                                {alert.image_alignment_status || 'N/A'}
                                            </span>
                                        </div>
                                        {alert.dispatched_at && (
                                            <div className="metadata-row">
                                                <span>Dispatched</span>
                                                <span>{formatTimestamp(alert.dispatched_at)}</span>
                                            </div>
                                        )}
                                        {alert.resolved_at && (
                                            <div className="metadata-row">
                                                <span>Archived</span>
                                                <span>{formatTimestamp(alert.resolved_at)}</span>
                                            </div>
                                        )}
                                    </div>

                                    <div className="popup-btns">
                                        {!ARCHIVED_STATUSES.has(alert.status) && alert.status !== 'DISPATCHED' && (
                                            <button className="btn-confirm" onClick={() => handleUpdateStatus(alert.id, 'DISPATCHED')}>
                                                DISPATCH
                                            </button>
                                        )}
                                        {!ARCHIVED_STATUSES.has(alert.status) && (
                                            <button className="btn-dismiss" onClick={() => handleUpdateStatus(alert.id, 'IGNORED')}>
                                                IGNORE
                                            </button>
                                        )}
                                        {alert.status === 'DISPATCHED' && (
                                            <span className="secondary-note">Dispatch confirmed.</span>
                                        )}
                                        {ARCHIVED_STATUSES.has(alert.status) && (
                                            <span className="secondary-note">Archived record.</span>
                                        )}
                                    </div>
                                </div>
                            </Popup>
                        </Marker>
                    );
                })}
            </MarkerClusterGroup>
        </MapContainer>
    );
};

export default Map;

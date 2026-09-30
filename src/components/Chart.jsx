import React from 'react'
import {
  PieChart,
  Pie,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'

const COLORS = {
  Accept: '#16a34a',
  accept: '#16a34a',
  Review: '#ea580c',
  review: '#ea580c',
  Reject: '#dc2626',
  reject: '#dc2626',
  primary: '#2563eb',
  LOT_001: '#2563eb',
  LOT_002: '#3b82f6',
  LOT_003: '#60a5fa',
  LOT_004: '#93c5fd',
  LOT_005: '#bfdbfe',
}

export default function Chart({ type = 'bar', data = [] }) {
  if (!data || data.length === 0) {
    return <div className="chart-empty">No data available</div>
  }

  if (type === 'pie') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${value}`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[entry.name] || COLORS.primary} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  if (type === 'bar') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="lot_id" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb' }} />
          <Legend />
          <Bar dataKey="accepted" fill={COLORS.Accept} />
          <Bar dataKey="review" fill={COLORS.Review} />
          <Bar dataKey="rejected" fill={COLORS.Reject} />
        </BarChart>
      </ResponsiveContainer>
    )
  }

  return <div className="chart-error">Unsupported chart type: {type}</div>
}

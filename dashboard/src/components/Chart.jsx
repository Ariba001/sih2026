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
  Accept: '#1b7f5a',
  ACCEPT: '#1b7f5a',
  Review: '#c47a00',
  REVIEW: '#c47a00',
  Reject: '#c0392b',
  REJECT: '#c0392b',
}

export default function Chart({ type = 'bar', data = [] }) {
  if (!data?.length) {
    return <div className="chart-empty">No chart data yet</div>
  }

  if (type === 'pie') {
    return (
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={95}
            paddingAngle={2}
            label={({ name, value }) => `${name} ${value}`}
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name] || '#0d9488'} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#d5dee8" />
        <XAxis dataKey="lot_id" tick={{ fontSize: 11, fill: '#3a4f6a' }} />
        <YAxis tick={{ fontSize: 11, fill: '#3a4f6a' }} />
        <Tooltip
          contentStyle={{
            background: '#fff',
            border: '1px solid #d5dee8',
            borderRadius: 8,
            fontFamily: 'IBM Plex Sans, sans-serif',
          }}
        />
        <Legend />
        <Bar dataKey="accepted" stackId="a" fill={COLORS.Accept} name="Accept" />
        <Bar dataKey="review" stackId="a" fill={COLORS.Review} name="Review" />
        <Bar dataKey="rejected" stackId="a" fill={COLORS.Reject} name="Reject" />
      </BarChart>
    </ResponsiveContainer>
  )
}

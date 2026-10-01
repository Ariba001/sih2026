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

/** SIH-aligned decision colours */
const COLORS = {
  Accept: '#138808',
  ACCEPT: '#138808',
  Review: '#E65100',
  REVIEW: '#E65100',
  Reject: '#C62828',
  REJECT: '#C62828',
}

const tooltipStyle = {
  background: '#fff',
  border: '1px solid #D5DEE8',
  borderRadius: 4,
  fontFamily: 'IBM Plex Sans, sans-serif',
  fontSize: 12,
  boxShadow: 'none',
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
            innerRadius={58}
            outerRadius={96}
            paddingAngle={1}
            stroke="#fff"
            strokeWidth={2}
            label={({ name, value }) => `${name} ${value}`}
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name] || '#0B3D6E'} />
            ))}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
        <XAxis dataKey="lot_id" tick={{ fontSize: 11, fill: '#4A5D73' }} axisLine={{ stroke: '#D5DEE8' }} />
        <YAxis tick={{ fontSize: 11, fill: '#4A5D73' }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend iconType="square" wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="accepted" stackId="a" fill={COLORS.Accept} name="Accept" />
        <Bar dataKey="review" stackId="a" fill={COLORS.Review} name="Review" />
        <Bar dataKey="rejected" stackId="a" fill={COLORS.Reject} name="Reject" />
      </BarChart>
    </ResponsiveContainer>
  )
}

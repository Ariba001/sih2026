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
  Accept: '#138808',
  ACCEPT: '#138808',
  Review: '#E65100',
  REVIEW: '#E65100',
  Reject: '#C62828',
  REJECT: '#C62828',
}

const tooltipStyle = {
  background: '#FFFFFF',
  border: '1px solid #C9D4E0',
  borderRadius: 2,
  fontFamily: 'IBM Plex Sans, sans-serif',
  fontSize: 12,
  color: '#0B3D6E',
  boxShadow: 'none',
  padding: '8px 10px',
}

function totalOf(data) {
  return data.reduce((sum, d) => sum + (Number(d.value) || 0), 0)
}

export default function Chart({ type = 'bar', data = [] }) {
  if (!data?.length) {
    return <div className="chart-empty">No chart data yet</div>
  }

  if (type === 'pie') {
    const total = totalOf(data)
    return (
      <div className="chart-shell">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="48%"
              innerRadius={62}
              outerRadius={92}
              paddingAngle={2}
              stroke="#fff"
              strokeWidth={2}
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={COLORS[entry.name] || '#0B3D6E'} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value, name) => [
                total ? `${value} (${Math.round((value / total) * 100)}%)` : value,
                name,
              ]}
            />
            <Legend
              verticalAlign="bottom"
              height={36}
              iconType="square"
              iconSize={8}
              formatter={(value) => {
                const row = data.find((d) => d.name === value)
                return `${value}  ${row?.value ?? 0}`
              }}
              wrapperStyle={{ fontSize: 12, color: '#3D5166', paddingTop: 4 }}
            />
            {total > 0 && (
              <text
                x="50%"
                y="46%"
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700, fill: '#0B3D6E' }}
              >
                <tspan x="50%" dy="-0.35em" fontSize="22">{total}</tspan>
                <tspan x="50%" dy="1.4em" fontSize="11" fill="#6B7C90" fontWeight={500} fontFamily="IBM Plex Sans, sans-serif">
                  screened
                </tspan>
              </text>
            )}
          </PieChart>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <div className="chart-shell">
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }} barCategoryGap="28%">
          <CartesianGrid strokeDasharray="0" stroke="#E8EEF4" vertical={false} />
          <XAxis
            dataKey="lot_id"
            tick={{ fontSize: 11, fill: '#3D5166' }}
            axisLine={{ stroke: '#C9D4E0' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11, fill: '#6B7C90' }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
          />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(11,61,110,0.04)' }} />
          <Legend
            iconType="square"
            iconSize={8}
            wrapperStyle={{ fontSize: 12, color: '#3D5166' }}
          />
          <Bar dataKey="accepted" stackId="a" fill={COLORS.Accept} name="Accept" radius={[0, 0, 0, 0]} />
          <Bar dataKey="review" stackId="a" fill={COLORS.Review} name="Review" />
          <Bar dataKey="rejected" stackId="a" fill={COLORS.Reject} name="Reject" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

import { createTheme } from '@mui/material/styles'

/** 全局 MUI 主题（T01 阶段为最小配置，后续可按需扩展） */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#7c4dff',
    },
    background: {
      default: '#0a0a0a',
    },
  },
  typography: {
    fontFamily: [
      '"PingFang SC"',
      '"Microsoft YaHei"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
  },
})

export default theme

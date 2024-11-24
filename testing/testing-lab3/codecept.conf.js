/** @type {CodeceptJS.MainConfig} */
exports.config = {
  tests: './*_test.js',
  output: './output',
  helpers: {
    Puppeteer: {
      url: 'https://goldapple.ru',
      show: true,
      restart: false,
      windowSize: '1920x1080',
      waitForNavigation: ['domcontentloaded', 'networkidle2'],
      waitForAction: 750
    }
  },
  include: {
    I: './steps_file.js',
    searchingPage: "./pages/searching.js",
  },
  name: 'testing-lab3'
}
const { I } = inject();

module.exports = {
  locators: {
    openButton: locate('button')
      .withClassAttr('ga-header__tab_type_search'),
    input: locate('input')
      .withAttr({ placeholder: 'хочу купить' }),
    submitButton: locate('button')
      .withAttr({ type: 'submit' })
      .withClassAttr('WsiT3 YvLEA l8SKP'),
    resultLabel: locate('span')
      .withClassAttr('YYut9')
  },

  searchOnSite(query, callback) {
    // make search request
    I.amOnPage('/');
    I.waitForNavigation();
    I.click(this.locators.openButton);
    I.fillField(this.locators.input, query);
    I.click(this.locators.submitButton);

    // check search result
    callback();

    // check search requests history contains query
    I.click(this.locators.openButton);
    I.see(query);
  }
}
Feature('Поиск товаров (артикул, название)');

Scenario('Поиск товара по названию (товар существует)', async ({ I, searchingPage }) => {
  const searchQuery = 'anastasia beverly hills lash brag';
  searchingPage.searchOnSite(searchQuery, () => {
    I.see('ANASTASIA BEVERLY HILLS', searchingPage.locators.resultLabelCSSClass);
  });
});

Scenario('Поиск товара по артикулу (товар существует)', async ({ I, searchingPage }) => {
  const searchQuery = '19000164782';
  searchingPage.searchOnSite(searchQuery, () => {
    I.see('ANASTASIA BEVERLY HILLS', searchingPage.locators. resultLabelCSSClass);
    I.see('1 продукт')
  });
});

Scenario('Поиск товара по названию (товар не существует)', async ({ I, searchingPage }) => {
  const searchQuery = 'абракадабра';
  searchingPage.searchMissingOnSite(searchQuery);
});

Scenario('Поиск товара по артикулу (товар не существует)', async ({ I, searchingPage }) => {
  const searchQuery = '39000164782';
  searchingPage.searchMissingOnSite(searchQuery);
});

Scenario('Недоступность кнопки отправки поискового запроса', async ({ I, searchingPage }) => {
  I.amOnPage('/');
  I.waitForNavigation();
  I.click(searchingPage.locators.openButton);
  I.seeElementInDOM(searchingPage.locators.submitButtonDisabled);
});

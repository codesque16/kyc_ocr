import React from 'react';
import { Route, Switch } from 'react-router-dom';
import About from './pages/About';
import Classify from './pages/Classify';
//import NotFound from './pages/NotFound';

export default ({ childProps }) =>
  <Switch>
    <Route path="/kyc_ocr/classify" exact component={Classify} props={childProps} />
    <Route path="/kyc_ocr/about" exact component={About} props={childProps} />
  </Switch>;

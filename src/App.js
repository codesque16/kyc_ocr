import { PropTypes } from 'prop-types';
import React, { Component } from 'react';
import { Link, withRouter } from 'react-router-dom';
import { Container, Nav, Navbar } from 'react-bootstrap';
import AlertDismissable from './components/AlertDismissable';
import Routes from './Routes';
import './App.css';
import openCV from 'react-opencvjs'


class App extends Component {

  constructor(props) {
    super(props);
    const reloadMsg = `
      New content is available.<br />
      Please <a href='javascript:location.reload();'>reload</a>.<br />
      <small>If reloading doesn't work, close all tabs/windows of this web application,
      and then reopen the application.</small>
    `;
    this.state = {
      showUpdateAlert: true,
      reloadMsg: reloadMsg
    };
  }

  dismissUpdateAlert = event => {
    this.setState({ showUpdateAlert: false });
  }

  componentDidMount() {
    openCV({
        onLoaded: () => {console.log(window.cv.getBuildInformation());console.log('open cv loaded')},
        onFailed: () => console.log('open cv failed to load'),
        version: '4.5.1'
    });
  }

  render() {
    return (
        <div className="App">
          <Container>
            <Navbar collapseOnSelect className="app-nav-bar" variant="dark" expand="lg">
              <Navbar.Brand href="/kyc_ocr">KYC</Navbar.Brand>
              <Navbar.Toggle aria-controls="basic-navbar-nav" />
              <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="">
                  <Link className="nav-link" to="/kyc_ocr/classify">Classify</Link>
                  <Link className="nav-link" to="/kyc_ocr/about">About</Link>
                </Nav>
              </Navbar.Collapse>
            </Navbar>
            { this.props.updateAvailable && this.state.showUpdateAlert &&
              <div style={{paddingTop: '10px'}}>
                <AlertDismissable
                  title=""
                  variant="info"
                  message={this.state.reloadMsg}
                  show={this.props.updateAvailable && this.state.showUpdateAlert}
                  onClose={this.dismissUpdateAlert} />
              </div>
            }
          </Container>
          <Container>
            <Routes />
          </Container>
        </div>
    );
  }
}

App.propTypes = {
  updateAvailable: PropTypes.bool.isRequired,
};

export default withRouter(App);

import 'antd/dist/antd.css';
import "./App.css"
import React, { useEffect, useState } from 'react';
import { Layout, Menu, Button } from 'antd';
import { useDropzone } from 'react-dropzone';
import Axios from 'axios';
import { BASE_URL } from './constant';

const { Header, Content } = Layout;

function App() {

  const thumbsContainer = {
    display: 'flex',
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 16
  };

  const thumb = {
    display: 'block',
    borderRadius: 2,
    border: '1px solid #eaeaea',
    marginBottom: 8,
    marginRight: 8,
    width: "100%",
    padding: 4,
    boxSizing: 'border-box',
    height: '464px',
  };

  const thumbInner = {
    display: 'flex',
    minWidth: 0,
    height: "100%"
  };

  const img = {
    display: 'block',
    width: '100%',

  };
  const resultdiv = {
    display: "flex",
    height: "79px",
    background: "#1890ff",
    textAlign: "center",
    flexDirection: "column",
    justifyContent: "center",
    fontSize: "20px"
  }

  const [files, setFiles, ,] = useState([]);
  const [response, setResponse] = useState(0);
  const { getRootProps, getInputProps } = useDropzone({
    accept: 'image/*',
    onDrop: acceptedFiles => {
      setFiles(acceptedFiles.map(file => Object.assign(file, {
        preview: URL.createObjectURL(file)
      })));
    },
    multiple: false
  });
  useEffect(() => () => {
    files.forEach(file => URL.revokeObjectURL(file.preview));
  }, [files]);


  const thumbs = files.map(file => (
    <div style={thumb} key={file.name}>
      <div style={thumbInner}>
        <img
          src={file.preview}
          style={img}
          alt="input file"
        />
      </div>
    </div>
  ));

  const upload = async () => {
    const data = Array.from(files);
    const formData = new FormData();
    formData.append('image', data[0]);
    let response = await Axios.request({
      url: `${BASE_URL}/predict`,
      method: "POST",
      data: formData
    });
    setResponse(response.data);

    console.log(response);
  }
  return (
    <div className="App">
      <header className="App-header">
        <Header className="header" style={{ marginBottom: "20px" }}>
          <div className="logo" />

          <Menu
            theme="dark"
            mode="horizontal"
            defaultSelectedKeys={['1']}
            style={{ lineHeight: '64px' }}
          >
            <Menu.Item key="1">MURA</Menu.Item>
          </Menu>

        </Header>
        <Content style={{ padding: '0 50px', margin: "0 auto", textAlign: "center" }}>

          <div style={{ background: '#fff', padding: 24, minHeight: 280, display: "flex" }}>
            <div style={{ flex: 1 }}>
              <section className="container">
                <div {...getRootProps({ className: 'dropzone' })}>
                  <input {...getInputProps()} />
                  <p>Drag and drop file here, or click to select files</p>
                </div>
                <aside style={thumbsContainer}>
                  {thumbs}
                </aside>
              </section>
              <Button type="dashed" shape="round" icon="download" size="default" onClick={upload}>
                Upload and process
            </Button>
            </div>
            <div style={{ flex: 1, padding: " 0 0 0 10px" }}>
              <div>


                <h3 style={{ width: "100%", color: "#fff" }}>{response ? <div style={resultdiv}>{response.label} </div> : null}</h3>

                {
                  response ?
                    <div style={{ ...thumb, marginTop: "18px" }}>
                      <div style={thumbInner}>
                        <img
                          src={`${BASE_URL}/image/${response.predict_image}`}
                          style={img}
                          alt="input file"
                        />
                      </div>
                    </div>
                    : null
                }
              </div>
            </div>
          </div>
        </Content>
      </header>
    </div >
  );
}

export default App;

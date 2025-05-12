import { useState, useEffect, useRef } from 'react';
import { Input, Button, Avatar, List } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import './App.css';
import { Content } from 'antd/es/layout/layout';

const { TextArea } = Input;

const App = () => {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false); // 控制是否正在等待 AI 回复
    const messagesEndRef = useRef(null); // 用于引用消息列表的底部

    // 自动滚动到底部的函数
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const postUserContent = async (userMessages, prompt) => {
        const url = 'http://localhost:10000/generate'; // Flask API 地址
        const userMessage = {
            chatHistory: userMessages,
            userPrompt: prompt
        }
        try {
            const response = await fetch(url, {
                method: 'POST', // 使用 POST 请求
                headers: {
                'Content-Type': 'application/json', // 请求体格式为 JSON
                },
                body: JSON.stringify(userMessage), // 将传入的用户内容转换为 JSON 字符串
            });
        
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
        
            const data = await response.json(); // 解析返回的 JSON 数据
            console.log(data);
            const awer = data['generated_text'];
            setMessages((prevMessages) => [
                ...prevMessages.filter((msg) => !msg.isTyping), 
                { text: `${awer}`, sender: 'ai' },
            ]);
            console.log(awer); // 打印响应数据
        } catch (error) {
            // 请求失败时，显示错误消息并恢复输入框
            setMessages((prevMessages) => [
                ...prevMessages.filter((msg) => !msg.isTyping), // 移除“生成中....”消息
                { text: `Error: ${error.message}`, sender: 'ai' }, // 显示错误消息
            ]);
            console.error('Error occurred:', error);
        } finally {
            // 无论请求成功或失败，都恢复输入框和按钮
            setIsLoading(false); // 解锁输入框和按钮
        }
        
      };

    // 监听 messages 的变化，每次更新时滚动到底部
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    
    

    const handleSend = () => {
        if (inputValue.trim() === '' || isLoading) return;

        // 添加用户消息，并确保更新状态基于最新的 messages
        setMessages((prevMessages) => {
            const updatedMessages = [...prevMessages, { text: inputValue, sender: 'user' }];
            // 添加“生成中....”消息
            updatedMessages.push({ text: 'generating....', sender: 'ai', isTyping: true });
            return updatedMessages;
        });
        setIsLoading(true); // 禁用输入框和按钮
        postUserContent(messages, inputValue);
        setIsLoading(false); // 恢复输入框和按钮
        setInputValue('');
    };

    

    return (
        <div className="chat-container">
            <div className="chat-messages">
                <List
                    dataSource={messages}
                    renderItem={(item) => (
                        <List.Item
                            style={{
                                justifyContent: item.sender === 'user' ? 'flex-end' : 'flex-start',
                            }}
                        >
                            <div className={`chat-bubble ${item.sender}`}>
                                {item.sender === 'ai' && <Avatar icon={<RobotOutlined />} />}
                                <div className={`bubble-content`}>
                                    {item.text}
                                </div>
                                {item.sender === 'user' && <Avatar icon={<UserOutlined />} />}
                            </div>
                        </List.Item>
                    )}
                />
                {/* 用于定位消息列表底部的空 div */}
                <div ref={messagesEndRef} />
            </div>
            <div className="chat-input">
                <TextArea
                    rows={2}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onPressEnter={handleSend}
                    disabled={isLoading} // 禁用输入框
                    autoSize={{ minRows: 2, maxRows: 6 }} // 自动调整高度，最小 2 行，最大 6 行
                    className="custom-textarea" // 添加自定义类名
                />
                <Button type="primary" onClick={handleSend} disabled={isLoading}>
                    Send
                </Button>
            </div>
        </div>
    );
};

export default App;
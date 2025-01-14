import { useState, useEffect, useRef } from 'react';
import { Input, Button, Avatar, List } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';
import './App.css';

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

    // 监听 messages 的变化，每次更新时滚动到底部
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = () => {
        if (inputValue.trim() === '' || isLoading) return;

        // 添加用户消息
        setMessages([...messages, { text: inputValue, sender: 'user' }]);
        setInputValue('');
        setIsLoading(true); // 禁用输入框和按钮

        // 添加“生成中....”消息
        setMessages((prevMessages) => [
            ...prevMessages,
            { text: '生成中....', sender: 'ai', isTyping: true },
        ]);

        // 模拟 AI 回复
        setTimeout(() => {
            setMessages((prevMessages) => [
                ...prevMessages.filter((msg) => !msg.isTyping), // 移除“生成中....”消息
                { text: `AI: ${inputValue}`, sender: 'ai' },
            ]);
            setIsLoading(false); // 恢复输入框和按钮
        }, 2000); // 模拟 2 秒延迟
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
                                <div className={`bubble-content ${item.isTyping ? 'typing' : ''}`}>
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
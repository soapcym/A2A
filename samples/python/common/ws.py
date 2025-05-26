import asyncio
import websockets
import threading
from typing import Dict, Optional, Union
import json
from my_utils.logger_util import logger
from llama_index.core.workflow import Context




# WebSocket连接管理器
def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class WebSocketManager:
    """WebSocket连接管理器，负责管理和维护WebSocket连接实例"""

    def __init__(self):
        # 修改类型注解 WebSocketClientProtocol
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.connection_lock = threading.Lock()
        self.connection_tasks: Dict[str, asyncio.Task] = {}

    async def connect_to_server(self, server_url: str, connection_id: str = None) -> str:
        """
        连接到指定的WebSocket服务器

        Args:
            server_url: WebSocket服务器URL
            connection_id: 连接标识符，如果为空则使用server_url作为标识符

        Returns:
            str: 连接标识符
        """
        if connection_id is None:
            connection_id = server_url

        with self.connection_lock:
            # 如果连接已存在且活跃，直接返回
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                # 修复：使用正确的方式检查连接状态
                if connection.close_code is None:
                    logger.info(f"连接 {connection_id} 已存在且活跃")
                    return connection_id
                else:
                    # 清理已关闭的连接
                    await self._cleanup_connection(connection_id)

        try:
            # 建立新的WebSocket连接
            logger.info(f"正在连接到WebSocket服务器: {server_url}")
            connection = await websockets.connect(server_url)

            with self.connection_lock:
                self.connections[connection_id] = connection
                # 启动心跳任务
                self.connection_tasks[connection_id] = asyncio.create_task(
                    self._keep_alive(connection_id)
                )

            logger.info(f"成功连接到WebSocket服务器，连接ID: {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"连接WebSocket服务器失败: {server_url}, 错误: {e}")
            raise

    async def send_message(self, connection_id: str, message: dict) -> bool:
        """
        通过指定连接发送消息

        Args:
            connection_id: 连接标识符
            message: 要发送的消息字典

        Returns:
            bool: 发送是否成功
        """
        connection = self.get_connection(connection_id)
        if not connection:
            logger.error(f"连接 {connection_id} 不存在")
            return False

        try:
            message_str = json.dumps(message, ensure_ascii=False)
            await connection.send(message_str)
            logger.debug(f"已发送消息到连接 {connection_id}: {message_str}")
            return True
        except Exception as e:
            logger.error(f"发送消息失败，连接ID: {connection_id}, 错误: {e}")
            await self._cleanup_connection(connection_id)
            return False

    async def receive_message(self, connection_id: str, timeout: float = 5.0) -> Optional[dict]:
        """
        从指定连接接收消息

        Args:
            connection_id: 连接标识符
            timeout: 超时时间（秒）

        Returns:
            dict: 接收到的消息字典，失败返回None
        """
        connection = self.get_connection(connection_id)
        if not connection:
            logger.error(f"连接 {connection_id} 不存在")
            return None

        try:
            message_str = await asyncio.wait_for(connection.recv(), timeout=timeout)
            message = json.loads(message_str)
            logger.debug(f"从连接 {connection_id} 接收到消息: {message_str}")
            return message
        except asyncio.TimeoutError:
            logger.warning(f"接收消息超时，连接ID: {connection_id}")
            return None
        except Exception as e:
            logger.error(f"接收消息失败，连接ID: {connection_id}, 错误: {e}")
            await self._cleanup_connection(connection_id)
            return None

    def get_connection(self, connection_id: str) -> Optional[websockets.WebSocketClientProtocol]:
        """
        获取指定的连接实例

        Args:
            connection_id: 连接标识符

        Returns:
            WebSocket连接实例或None
        """
        with self.connection_lock:
            return self.connections.get(connection_id)

    async def close_connection(self, connection_id: str):
        """
        关闭指定的连接

        Args:
            connection_id: 连接标识符
        """
        logger.info(f"正在关闭连接: {connection_id}")
        await self._cleanup_connection(connection_id)

    async def close_all_connections(self):
        """关闭所有连接"""
        logger.info("正在关闭所有WebSocket连接")
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            await self._cleanup_connection(connection_id)

    def get_active_connections(self) -> Dict[str, bool]:
        """
        获取所有活跃连接的状态

        Returns:
            dict: 连接ID到状态的映射
        """
        status = {}
        with self.connection_lock:
            for conn_id, connection in self.connections.items():
                # 修复：使用正确的方式检查连接状态
                status[conn_id] = connection.close_code is None
        return status

    async def _keep_alive(self, connection_id: str):
        """
        保持连接活跃的心跳任务

        Args:
            connection_id: 连接标识符
        """
        try:
            while True:
                connection = self.get_connection(connection_id)
                # 修复：使用正确的方式检查连接状态
                if not connection or connection.close_code is not None:
                    break

                # 发送心跳ping
                try:
                    await connection.ping()
                    await asyncio.sleep(30)  # 每30秒发送一次心跳
                except Exception as e:
                    logger.warning(f"心跳失败，连接ID: {connection_id}, 错误: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug(f"心跳任务被取消，连接ID: {connection_id}")
        except Exception as e:
            logger.error(f"心跳任务异常，连接ID: {connection_id}, 错误: {e}")

        # 清理连接
        await self._cleanup_connection(connection_id)

    async def _cleanup_connection(self, connection_id: str):
        """
        清理指定连接的资源

        Args:
            connection_id: 连接标识符
        """
        with self.connection_lock:
            # 取消心跳任务
            if connection_id in self.connection_tasks:
                task = self.connection_tasks.pop(connection_id)
                if not task.done():
                    task.cancel()

            # 关闭连接
            if connection_id in self.connections:
                connection = self.connections.pop(connection_id)
                # 修复：使用正确的方式检查和关闭连接
                if connection.close_code is None:
                    await connection.close()

        logger.info(f"已清理连接: {connection_id}")


# 全局WebSocket管理器实例
websocket_manager = WebSocketManager()


# 在现有代码中添加WebSocket相关功能
async def send_websocket_command(ctx: Context, command: dict) -> dict:
    """
    通过WebSocket发送命令到远程服务器

    Args:
        ctx: 工作流上下文
        command: 要发送的命令字典

    Returns:
        dict: 服务器响应
    """

    logger.info(f"send_websocket_command begin")


    # 获取或创建WebSocket连接
    server_url = f"ws://118.178.131.169:8081"  # 根据实际情况修改URL

    try:
        connection_id = await websocket_manager.connect_to_server(server_url)

        # 发送命令
        success = await websocket_manager.send_message(connection_id, command)
        if not success:
            return {"error": "发送命令失败"}

        # 接收响应
        response = await websocket_manager.receive_message(connection_id, timeout=10.0)
        if response is None:
            return {"error": "接收响应超时"}

        return response

    except Exception as e:
        logger.error(f"WebSocket命令执行失败: {e}")
        return {"error": str(e)}
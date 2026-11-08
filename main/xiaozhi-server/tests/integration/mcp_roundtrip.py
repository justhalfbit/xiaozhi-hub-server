"""Local MCP handshake/tool-call smoke test, without cloud services or API keys.

Run from xiaozhi-server: python tests/integration/mcp_roundtrip.py
Standalone because the ordinary pytest conftest replaces HTTP clients with mocks.
"""
import asyncio
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.admin import mcp_settings
from core.providers.tools.server_mcp.mcp_manager import ServerMCPManager


def serve(port):
    from mcp.server.fastmcp import FastMCP
    app = FastMCP('Fixture', host='127.0.0.1', port=port, stateless_http=True, json_response=True)

    @app.tool()
    def fixture_echo(text: str) -> str:
        """Echo test text without side effects."""
        return text

    app.run(transport='streamable-http')


async def check(port, directory):
    path = directory / '.mcp_server_settings.json'
    mcp_settings.save(path, 'fixture', {'url': f'http://127.0.0.1:{port}/mcp', 'transport': 'streamable-http'})
    manager = ServerMCPManager(SimpleNamespace())
    manager.config_path = str(path)
    try:
        await manager.initialize_servers()
        tools = manager.get_all_tools()
        name = next(t['function']['name'] for t in tools if 'fixture_echo' in t['function']['name'])
        result = await manager.execute_tool(name, {'text': '繁體中文 MCP 測試'})
        assert not result.isError
        assert any(c.text == '繁體中文 MCP 測試' for c in result.content if c.type == 'text')
    finally:
        await manager.cleanup_all()


def main():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='mcp-roundtrip-') as tmp:
        with open(Path(tmp) / 'server.log', 'w') as log:
            proc = subprocess.Popen([sys.executable, __file__, '--server', str(port)], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=.2):
                            break
                    except OSError:
                        if proc.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('MCP fixture failed to start')
                        time.sleep(.1)
                asyncio.run(check(port, Path(tmp)))
                print('PASS: saved MCP config -> initialize -> tools/list -> tools/call -> cleanup')
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--server':
        serve(int(sys.argv[2]))
    else:
        main()

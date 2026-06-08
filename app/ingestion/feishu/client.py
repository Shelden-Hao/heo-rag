import httpx
from app.config import settings


BASE_URL = "https://open.feishu.cn/open-apis"


class FeishuClient:
    def __init__(self):
        self.app_id = settings.feishu_app_id
        self.app_secret = settings.feishu_app_secret
        self._token: str | None = None

    def _get_tenant_token(self) -> str:
        if self._token:
            return self._token
        resp = httpx.post(
            f"{BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise ValueError(f"Feishu auth failed: {data}")
        self._token = data["tenant_access_token"]
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_tenant_token()}"}

    def list_documents(self, folder_token: str) -> list[dict]:
        files = []
        page_token = None
        while True:
            params = {"folder_token": folder_token, "page_size": 50}
            if page_token:
                params["page_token"] = page_token
            resp = httpx.get(
                f"{BASE_URL}/drive/v1/files",
                headers=self._headers(),
                params=params,
            )
            data = resp.json()
            code = data.get("code", -1)
            if code != 0:
                msg = data.get("msg", resp.text)
                raise ValueError(
                    f"飞书 API 错误 (code={code}): {msg}\n"
                    f"请检查：1) 应用已发布 2) 已添加 drive:drive:readonly 权限 3) folder_token 正确"
                )
            for f in data.get("data", {}).get("files", []):
                if f.get("type") == "docx":
                    files.append({
                        "token": f["token"],
                        "name": f["name"],
                        "type": f["type"],
                    })
            if not data.get("data", {}).get("has_more"):
                break
            page_token = data["data"].get("page_token")
        return files

    def get_document_content(self, document_id: str) -> dict:
        resp = httpx.get(
            f"{BASE_URL}/docx/v1/documents/{document_id}/raw_content",
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise ValueError(f"Get document failed: {data}")
        return data["data"]["content"]

    def get_document_blocks(self, document_id: str) -> list[dict]:
        blocks = []
        page_token = None
        while True:
            params = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            resp = httpx.get(
                f"{BASE_URL}/docx/v1/documents/{document_id}/blocks",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise ValueError(f"Get blocks failed: {data}")
            blocks.extend(data.get("data", {}).get("items", []))
            if not data.get("data", {}).get("has_more"):
                break
            page_token = data["data"].get("page_token")
        return blocks


feishu_client = FeishuClient()

# Implementation Plan: Orchestrator Refactor

## Overview

`AgentOrchestrator` sınıfının bağımlılık enjeksiyonu, async KnowledgeBase yüklemesi ve Protocol soyutlaması ile yeniden yapılandırılması. Uygulama sırası: Protocol tanımları → OrchestratorDependencies → __init__ refactor + from_settings → KnowledgeBase.async_load → AgentPlanner → __init__.py temizliği → service.py → PBT testleri → regresyon doğrulaması.

## Tasks

- [x] 1. Protocol tanımlarını ve OrchestratorDependencies dataclass'ını orchestrator.py'ye ekle
  - `orchestrator.py` dosyasının en üstüne (mevcut import bloğunun hemen altına) `Policy` ve `ReviewVerifierProtocol` Protocol sınıflarını ekle
  - `Policy`: `apply(self, parsed: ParsedAnswer) -> ParsedAnswer` metodunu tanımla; `@runtime_checkable` dekoratörü ekle
  - `ReviewVerifierProtocol`: `verify(self, workspace_path: Path, parsed: ParsedAnswer, selected_context: list[FileContext]) -> ParsedAnswer` metodunu tanımla
  - `OrchestratorDependencies` dataclass'ını tanımla; tüm 15 alanı (`scanner`, `ranker`, `reader`, `llm_provider`, `planner`, `response_parser`, `suggestion_policy`, `patch_preview_policy`, `apply_policy`, `review_verifier`, `session_writer`, `patch_writer`, `apply_log_writer`, `apply_engine`, `knowledge_base`) ve opsiyonel `settings: Optional[Settings] = field(default=None)` alanını içer
  - `dataclasses` ve `typing` importlarını ekle
  - _Requirements: 1.1, 1.3, 3.1, 3.3, 3.5_

- [x] 2. AgentOrchestrator.__init__ metodunu geriye dönük uyumlu şekilde refactor et
  - [x] 2.1 `__init__` imzasını `deps_or_settings: OrchestratorDependencies | Settings | None = None, *, deps: OrchestratorDependencies | None = None` olarak güncelle
  - `isinstance(deps_or_settings, Settings)` dalında `_build_deps_sync(settings)` yardımcı fonksiyonunu çağır (senkron yol — mevcut `AgentOrchestrator(settings)` kullanımını kırmaz)
  - `isinstance(deps_or_settings, OrchestratorDependencies)` dalında doğrudan kullan
  - `deps` keyword argümanı verilmişse onu kullan
  - Hiçbiri verilmemişse `TypeError` fırlat
  - `self.deps = resolved_deps` ata; ardından tüm kısayol referansları (`self.settings`, `self.scanner`, ..., `self.knowledge_base`) `self.deps.*` üzerinden ata
  - `self.logger` için `deps.settings` varsa `deps.settings.logs_dir / "editor-agent.log"`, yoksa `Path("editor-agent.log")` kullan
  - _Requirements: 1.1, 1.4, 6.4_

  - [x] 2.2 `_build_deps_sync(settings: Settings) -> OrchestratorDependencies` yardımcı fonksiyonunu ekle
  - Mevcut `__init__` içindeki tüm somutlaştırma mantığını bu fonksiyona taşı (`WorkspaceScanner(settings)`, `KnowledgeBase.load(...)`, vb.)
  - Senkron `KnowledgeBase.load()` kullanımını koru (bu yol yalnızca eski `AgentOrchestrator(settings)` çağrıları için)
  - `OrchestratorDependencies(...)` döndür
  - _Requirements: 1.2, 1.5, 6.4_

  - [x] 2.3 `from_settings` async classmethod'unu ekle
  - `@classmethod async def from_settings(cls, settings: Settings) -> "AgentOrchestrator"` imzasıyla tanımla
  - `KnowledgeBase.async_load(settings.state_dir / "knowledge_base.json")` çağrısını `await` ile yap
  - `OrchestratorDependencies(...)` oluştur ve `cls(deps)` döndür
  - _Requirements: 1.2, 1.5, 2.3_

- [x] 3. KnowledgeBase.async_load() metodunu knowledge.py'ye ekle
  - `knowledge.py` dosyasına `_read_json_file(path: Path) -> dict` modül düzeyinde yardımcı fonksiyonu ekle (thread içinde çalışacak senkron okuyucu)
  - `KnowledgeBase` sınıfına `@classmethod async def async_load(cls, path: Path) -> "KnowledgeBase"` metodunu ekle
  - `asyncio.to_thread(_read_json_file, path)` ile dosya okumayı event loop dışına taşı
  - `FileNotFoundError`, `json.JSONDecodeError`, `ValidationError`, `OSError` istisnalarını yakala; boş `cls()` döndür ve `logger.warning(...)` ile logla
  - Mevcut senkron `load()` metodunu koru; docstring'ine "Senkron bağlamlar için. Async bağlamlarda `async_load()` kullanın." notu ekle
  - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [ ]* 3.1 Property 2 için hypothesis testi yaz: async_load hata toleransı
    - **Property 2: async_load Hata Toleransı**
    - `hypothesis.strategies.text()` ile rastgele geçersiz JSON string'leri geçici dosyaya yaz; `async_load` çağır
    - `hypothesis.strategies.builds(Path, st.text())` ile var olmayan path'ler oluştur; `async_load` çağır
    - Her iki durumda da `KnowledgeBase` örneği döndüğünü, exception fırlatılmadığını doğrula
    - **Validates: Requirements 2.5**

- [x] 4. AgentPlanner'ı güncelle
  - `planner.py` dosyasının en üstüne modül düzeyinde docstring ekle (sınıfın sorumluluğunu, `plan_label` ve `plan_steps` metodlarının amacını açıkla)
  - `plan_label` metodunu güncelle: `ANALYZE`, `ASK`, `SUGGEST` dallarını koru; bilinmeyen mod için `ValueError` yerine `"Unknown"` döndür (default `return "Unknown"`)
  - `plan_steps(mode: AgentMode) -> list[str]` statik metodunu ekle; şimdilik `return []` döndür; docstring ile gelecekteki genişletme amacını belirt
  - _Requirements: 4.1, 4.3, 4.4_

  - [ ]* 4.1 Property 4 için hypothesis testi yaz: plan_label bilinmeyen mod toleransı
    - **Property 4: plan_label Bilinmeyen Mod Toleransı**
    - `hypothesis.strategies.text()` ile rastgele string'ler üret; geçerli `AgentMode` değerlerini (`filter` ile) hariç tut
    - `AgentPlanner.plan_label()` çağrısının `"Unknown"` döndürdüğünü ve exception fırlatmadığını doğrula
    - **Validates: Requirements 4.3**

- [x] 5. agent_core/agent/__init__.py'yi güncelle
  - Mevcut içeriği (`"""Agent orchestration layer."""`) koru
  - `AgentOrchestrator` ve `OrchestratorDependencies` sınıflarını `orchestrator.py`'den import et ve `__all__` listesine ekle
  - Diğer iç modülleri (`planner`, `suggester`, `patch_preview`, `apply_mode`, `review_verifier`, `response_parser`) doğrudan expose etme
  - _Requirements: 5.2, 5.3, 5.4_

- [x] 6. Checkpoint — Temel yapı doğrulaması
  - `python -c "from agent_core.agent import AgentOrchestrator, OrchestratorDependencies"` komutunun `ImportError` veya `CircularImportError` fırlatmadan çalıştığını doğrula
  - `python -c "from agent_core.agent.planner import AgentPlanner; print(AgentPlanner.plan_label('unknown'))"` çıktısının `"Unknown"` olduğunu doğrula
  - Tüm testlerin geçtiğini doğrula; sorun varsa kullanıcıya sor.

- [x] 7. service.py'yi from_settings kullanacak şekilde güncelle
  - `service.py` içindeki `get_orchestrator()` fonksiyonunu güncelle: `_orchestrator = AgentOrchestrator(settings=Settings.load())` satırını `_orchestrator = await AgentOrchestrator.from_settings(Settings.load())` ile değiştir
  - `get_orchestrator()` fonksiyonunu `async def get_orchestrator()` olarak güncelle
  - `get_orchestrator()` çağrılarını `run_agent` ve `stream_agent` içinde `await get_orchestrator()` olarak güncelle
  - `build_orchestrator()` alias fonksiyonunu da async yap veya kaldır (geriye dönük uyumluluk gerekiyorsa koru)
  - _Requirements: 1.2, 2.3, 6.1, 6.2_

- [x] 8. PBT testlerini tests/ dizinine ekle
  - [x] 8.1 `tests/test_orchestrator_pbt.py` dosyasını oluştur; `hypothesis` importlarını ve temel fixture'ları ekle
    - `pytest`, `hypothesis`, `asyncio`, `unittest.mock` importlarını ekle
    - `mock_deps()` fixture'ı oluştur: tüm `OrchestratorDependencies` alanlarını `MagicMock()` ile doldur; `settings` alanını `None` yap
    - _Requirements: 1.3, 1.4_

  - [ ]* 8.2 Property 1 testini yaz: Settings'e bağımsız çalışma
    - **Property 1: Settings'e Bağımsız Çalışma**
    - `@given(st.just(mock_deps()))` ile `AgentOrchestrator(deps=mock_deps)` başlat
    - `orchestrator.settings` alanının `None` olduğunu doğrula
    - `run()` çağrısı sırasında `Settings` nesnesinin herhangi bir alanına erişilmediğini doğrula (mock üzerinden)
    - **Validates: Requirements 1.4**

  - [ ]* 8.3 Property 3 testini yaz: Policy enjeksiyonu
    - **Property 3: Policy Enjeksiyonu**
    - `@given(st.builds(...))` ile `apply(parsed) -> parsed` imzasını karşılayan stub policy nesneleri üret
    - Stub'ı `OrchestratorDependencies.suggestion_policy` olarak geçir
    - `SUGGEST` modunda `run()` çağrısının stub'ın `apply` metodunu çağırdığını doğrula
    - **Validates: Requirements 3.4**

  - [ ]* 8.4 Property 5 testini yaz: Public API imzası korunması
    - **Property 5: Public API İmzası Korunması**
    - `@given(st.sampled_from(AgentMode), st.text(min_size=1))` ile geçerli mod ve user_input kombinasyonları üret
    - `run()` dönüş tipinin `SessionRecord` olduğunu doğrula
    - `stream_run()` dönüş değerinin `AsyncIterable` olduğunu doğrula
    - **Validates: Requirements 6.1, 6.2**

- [x] 9. Regresyon doğrulaması
  - `tests/test_agent_core.py` dosyasında herhangi bir değişiklik yapmadan `pytest tests/test_agent_core.py -v` komutunu çalıştır
  - Tüm testlerin geçtiğini doğrula; `AgentOrchestrator(settings)` çağrısının `TypeError` fırlatmadığını teyit et
  - `from agent_core.agent import AgentOrchestrator` import'unun döngüsel import olmadan çalıştığını doğrula
  - _Requirements: 6.4, 6.5, 5.3_

- [x] 10. Final checkpoint — Tüm testler ve import doğrulaması
  - `pytest tests/ -v` ile tüm test suite'ini çalıştır
  - Tüm testlerin geçtiğini doğrula; sorun varsa kullanıcıya sor.

## Notes

- `*` ile işaretli sub-task'lar opsiyoneldir; MVP için atlanabilir
- Her task ilgili requirements'a referans içerir
- Task 6 ve 10 checkpoint'tir; sorun varsa durulup kullanıcıya sorulur
- Property testleri `hypothesis` kütüphanesi ile yazılır; minimum 100 iterasyon
- `test_agent_core.py` dosyası hiçbir koşulda değiştirilmez (Requirement 6.5)
- `service.py` güncellemesi (Task 7) `from_settings` async olduğu için `get_orchestrator()` fonksiyonunu async yapar; bu değişiklik `api.py` çağrı noktalarını da etkiler — Task 7 sırasında `api.py` incelenmeli

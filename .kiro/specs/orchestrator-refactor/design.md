# Design Document: Orchestrator Refactor

## Overview

Bu tasarım, `AgentOrchestrator` sınıfının mimari refactor'ını kapsar. Mevcut yapıda orchestrator, `__init__` içinde 14+ bağımlılığı doğrudan somutlaştırmakta; bu durum birim testlerini zorlaştırmakta, policy sınıflarının değiştirilmesini engellemekte ve `KnowledgeBase.load()` çağrısının async event loop'u bloke etmesine yol açmaktadır.

Refactor'ın hedefleri:

1. **Bağımlılık enjeksiyonu**: `OrchestratorDependencies` dataclass'ı ile tüm bağımlılıkları dışarıdan enjekte edilebilir hale getirmek.
2. **Async IO**: `KnowledgeBase.async_load()` ile dosya okumayı event loop dışına taşımak.
3. **Protocol soyutlaması**: Policy ve ReviewVerifier sınıflarını `typing.Protocol` ile soyutlamak.
4. **AgentPlanner netleştirmesi**: Docstring, hata toleransı ve iskelet metot eklemek.
5. **Döngüsel import giderme**: `agent_core.agent.__init__` ihracat listesini daraltmak.
6. **Geriye dönük uyumluluk**: Mevcut public API ve test dosyası değiştirilmeden çalışmaya devam etmeli.

Kapsam dışı: `WorkspaceScanner`, `WorkspaceRanker`, `WorkspaceReader`, `LLMProvider`, `NvidiaProvider`, `models.py`, `prompts.py`, `server/` paketi, `ApplyEngine`, `SessionWriter`, `PatchProposalWriter`, `ApplyLogWriter`.

---

## Architecture

### Mevcut Yapı (Before)

```
AgentOrchestrator.__init__(settings: Settings)
  ├── WorkspaceScanner(settings)       # doğrudan somutlaştırma
  ├── WorkspaceRanker(settings)
  ├── WorkspaceReader(settings)
  ├── get_llm_provider(settings)
  ├── AgentPlanner()
  ├── ResponseParser()
  ├── SuggestionPolicy()               # değiştirilemez
  ├── PatchPreviewPolicy()             # değiştirilemez
  ├── ApplyModePolicy()                # değiştirilemez
  ├── ReviewVerifier(settings)
  ├── SessionWriter(settings)
  ├── PatchProposalWriter(settings)
  ├── ApplyLogWriter(settings)
  ├── ApplyEngine(settings)
  └── KnowledgeBase.load(...)          # senkron, event loop'u bloke eder
```

### Hedef Yapı (After)

```
OrchestratorDependencies (dataclass)
  ├── scanner: WorkspaceScanner
  ├── ranker: WorkspaceRanker
  ├── reader: WorkspaceReader
  ├── llm_provider: LLMProvider
  ├── planner: AgentPlanner
  ├── response_parser: ResponseParser
  ├── suggestion_policy: SuggestionPolicy (Policy protokolü)
  ├── patch_preview_policy: PatchPreviewPolicy (Policy protokolü)
  ├── apply_policy: ApplyModePolicy (Policy protokolü)
  ├── review_verifier: ReviewVerifier (ReviewVerifierProtocol)
  ├── session_writer: SessionWriter
  ├── patch_writer: PatchProposalWriter
  ├── apply_log_writer: ApplyLogWriter
  ├── apply_engine: ApplyEngine
  ├── knowledge_base: KnowledgeBase
  └── settings: Settings | None        # opsiyonel, geriye dönük uyumluluk

AgentOrchestrator.__init__(deps: OrchestratorDependencies)
  └── self.deps = deps                 # tüm bağımlılıklar dışarıdan gelir

AgentOrchestrator.from_settings(settings: Settings) -> AgentOrchestrator  [classmethod, async]
  └── KnowledgeBase.async_load(...)    # await ile çağrılır
```

### Modül Organizasyonu

```
agent_core/agent/
  ├── __init__.py          # Sadece AgentOrchestrator, OrchestratorDependencies export eder
  ├── orchestrator.py      # AgentOrchestrator + OrchestratorDependencies + Protocol tanımları
  ├── planner.py           # AgentPlanner (docstring + plan_steps eklenir)
  ├── suggester.py         # SuggestionPolicy (değişmez)
  ├── patch_preview.py     # PatchPreviewPolicy (değişmez)
  ├── apply_mode.py        # ApplyModePolicy (değişmez)
  ├── review_verifier.py   # ReviewVerifier (değişmez)
  └── response_parser.py   # ResponseParser (değişmez)
```

`OrchestratorDependencies` ve Protocol tanımları `orchestrator.py` içinde tanımlanır. Ayrı bir `deps.py` modülü oluşturulması opsiyoneldir; döngüsel import riski yoksa `orchestrator.py` içinde tutmak yeterlidir.

---

## Components and Interfaces

### 1. `Policy` Protokolü

```python
from typing import Protocol, runtime_checkable
from agent_core.models import ParsedAnswer

@runtime_checkable
class Policy(Protocol):
    def apply(self, parsed: ParsedAnswer) -> ParsedAnswer:
        ...
```

- `SuggestionPolicy`, `PatchPreviewPolicy`, `ApplyModePolicy` sınıfları bu protokolü yapısal olarak karşılar (miras almak zorunda değil).
- `runtime_checkable` ile `isinstance(obj, Policy)` kontrolü yapılabilir.

### 2. `ReviewVerifierProtocol` Protokolü

```python
from typing import Protocol
from pathlib import Path
from agent_core.models import FileContext, ParsedAnswer

class ReviewVerifierProtocol(Protocol):
    def verify(
        self,
        workspace_path: Path,
        parsed: ParsedAnswer,
        selected_context: list[FileContext],
    ) -> ParsedAnswer:
        ...
```

### 3. `OrchestratorDependencies` Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional
from agent_core.config import Settings

@dataclass
class OrchestratorDependencies:
    scanner: WorkspaceScanner
    ranker: WorkspaceRanker
    reader: WorkspaceReader
    llm_provider: LLMProvider
    planner: AgentPlanner
    response_parser: ResponseParser
    suggestion_policy: Policy
    patch_preview_policy: Policy
    apply_policy: Policy
    review_verifier: ReviewVerifierProtocol
    session_writer: SessionWriter
    patch_writer: PatchProposalWriter
    apply_log_writer: ApplyLogWriter
    apply_engine: ApplyEngine
    knowledge_base: KnowledgeBase
    settings: Optional[Settings] = field(default=None)
```

**Tasarım kararı**: `dataclass` tercih edildi çünkü bu nesne saf bir veri taşıyıcısıdır; Pydantic validasyonu gereksizdir ve `dataclass` daha hafiftir. `settings` alanı opsiyoneldir; `run()` içinde `self.settings.nvidia_model` gibi erişimler `self.deps.settings` üzerinden yapılır.

### 4. `AgentOrchestrator` Değişiklikleri

```python
class AgentOrchestrator:
    def __init__(self, deps: OrchestratorDependencies) -> None:
        self.deps = deps
        # Kısayol referanslar (geriye dönük uyumluluk için)
        self.settings = deps.settings
        self.scanner = deps.scanner
        self.ranker = deps.ranker
        self.reader = deps.reader
        self.client = deps.llm_provider
        self.planner = deps.planner
        self.response_parser = deps.response_parser
        self.suggestion_policy = deps.suggestion_policy
        self.patch_preview_policy = deps.patch_preview_policy
        self.apply_policy = deps.apply_policy
        self.review_verifier = deps.review_verifier
        self.session_writer = deps.session_writer
        self.patch_writer = deps.patch_writer
        self.apply_log_writer = deps.apply_log_writer
        self.apply_engine = deps.apply_engine
        self.knowledge_base = deps.knowledge_base
        self.logger = configure_logger(
            deps.settings.logs_dir / "editor-agent.log"
            if deps.settings else Path("editor-agent.log")
        )

    @classmethod
    async def from_settings(cls, settings: Settings) -> "AgentOrchestrator":
        """Fabrika metodu: Settings'ten varsayılan bağımlılıkları oluşturur."""
        knowledge_base = await KnowledgeBase.async_load(
            settings.state_dir / "knowledge_base.json"
        )
        deps = OrchestratorDependencies(
            scanner=WorkspaceScanner(settings),
            ranker=WorkspaceRanker(settings),
            reader=WorkspaceReader(settings),
            llm_provider=get_llm_provider(settings),
            planner=AgentPlanner(),
            response_parser=ResponseParser(),
            suggestion_policy=SuggestionPolicy(),
            patch_preview_policy=PatchPreviewPolicy(),
            apply_policy=ApplyModePolicy(),
            review_verifier=ReviewVerifier(settings),
            session_writer=SessionWriter(settings),
            patch_writer=PatchProposalWriter(settings),
            apply_log_writer=ApplyLogWriter(settings),
            apply_engine=ApplyEngine(settings),
            knowledge_base=knowledge_base,
            settings=settings,
        )
        return cls(deps)
```

**Geriye dönük uyumluluk**: Mevcut `AgentOrchestrator(settings)` çağrıları `TypeError` fırlatır çünkü `__init__` artık `deps` bekler. Bu sorunu çözmek için `__init__` içinde `settings` parametresini de kabul eden bir uyumluluk katmanı eklenir:

```python
def __init__(
    self,
    deps_or_settings: OrchestratorDependencies | Settings | None = None,
    *,
    deps: OrchestratorDependencies | None = None,
) -> None:
    if isinstance(deps_or_settings, Settings):
        # Eski kullanım: AgentOrchestrator(settings)
        # Senkron yol — async_load yerine senkron load kullanılır
        settings = deps_or_settings
        resolved_deps = _build_deps_sync(settings)
    elif isinstance(deps_or_settings, OrchestratorDependencies):
        resolved_deps = deps_or_settings
    elif deps is not None:
        resolved_deps = deps
    else:
        raise TypeError("OrchestratorDependencies veya Settings gereklidir.")
    ...
```

Bu yaklaşım mevcut `AgentOrchestrator(settings)` çağrılarını kırmaz ve `test_agent_core.py` dosyasının değiştirilmesini gerektirmez.

### 5. `KnowledgeBase.async_load()`

```python
@classmethod
async def async_load(cls, path: Path) -> "KnowledgeBase":
    """Async dosya okuma — event loop'u bloke etmez."""
    import asyncio
    try:
        data = await asyncio.to_thread(_read_json_file, path)
        return cls.model_validate(data)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Failed to async_load knowledge base from %s: %s", path, exc)
        return cls()

def _read_json_file(path: Path) -> dict:
    """Thread içinde çalışan senkron yardımcı."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

Mevcut senkron `load()` metodu korunur; yalnızca async olmayan bağlamlarda kullanılmak üzere docstring'e not eklenir.

### 6. `AgentPlanner` Değişiklikleri

```python
"""
AgentPlanner — Ajan çalışma modunu etiket ve adım listesine dönüştürür.

Sorumluluklar:
  - plan_label(mode): AgentMode değerini insan-okunabilir etikete çevirir.
  - plan_steps(mode): Gelecekteki genişletmeler için adım listesi iskeletini sağlar.

Bu sınıf durumsuz (stateless) bir yardımcıdır; orchestrator tarafından
çalışma döngüsünün başında çağrılır.
"""

class AgentPlanner:
    @staticmethod
    def plan_label(mode: AgentMode) -> str:
        if mode == AgentMode.ANALYZE:
            return "Analyze"
        if mode == AgentMode.ASK:
            return "Ask"
        if mode == AgentMode.SUGGEST:
            return "Suggest"
        # Bilinmeyen mod: ValueError yerine "Unknown" döndür
        return "Unknown"

    @staticmethod
    def plan_steps(mode: AgentMode) -> list[str]:
        """Gelecekteki genişletmeler için iskelet. Şimdilik boş liste döndürür."""
        return []
```

---

## Data Models

### `OrchestratorDependencies` Alanları

| Alan | Tip | Açıklama |
|------|-----|----------|
| `scanner` | `WorkspaceScanner` | Workspace dosya tarama |
| `ranker` | `WorkspaceRanker` | Dosya sıralama/önceliklendirme |
| `reader` | `WorkspaceReader` | Dosya içeriği okuma |
| `llm_provider` | `LLMProvider` | LLM arka uç soyutlaması |
| `planner` | `AgentPlanner` | Mod etiket/adım planlayıcı |
| `response_parser` | `ResponseParser` | LLM yanıt ayrıştırıcı |
| `suggestion_policy` | `Policy` | SUGGEST modu dönüşüm politikası |
| `patch_preview_policy` | `Policy` | PATCH_PREVIEW modu dönüşüm politikası |
| `apply_policy` | `Policy` | APPLY modu dönüşüm politikası |
| `review_verifier` | `ReviewVerifierProtocol` | REVIEW modu doğrulayıcı |
| `session_writer` | `SessionWriter` | Oturum kayıt yazıcı |
| `patch_writer` | `PatchProposalWriter` | Patch önerisi yazıcı |
| `apply_log_writer` | `ApplyLogWriter` | Uygulama logu yazıcı |
| `apply_engine` | `ApplyEngine` | Dosya operasyonu uygulayıcı |
| `knowledge_base` | `KnowledgeBase` | Geçmiş çözüm deposu |
| `settings` | `Settings \| None` | Opsiyonel yapılandırma (geriye dönük uyumluluk) |

### Import Bağımlılık Grafiği (Hedef)

```
orchestrator.py
  ├── agent_core.models          (ParsedAnswer, SessionRecord, ...)
  ├── agent_core.config          (Settings)
  ├── agent_core.knowledge       (KnowledgeBase)
  ├── agent_core.llm             (get_llm_provider, LLMProvider)
  ├── agent_core.output.writers  (SessionWriter, ...)
  ├── agent_core.tools.*         (ApplyEngine, ...)
  ├── agent_core.workspace.*     (WorkspaceScanner, ...)
  ├── agent_core.prompts         (build_prompt)
  ├── .planner                   (AgentPlanner)
  ├── .response_parser           (ResponseParser)
  ├── .suggester                 (SuggestionPolicy)
  ├── .patch_preview             (PatchPreviewPolicy)
  ├── .apply_mode                (ApplyModePolicy)
  └── .review_verifier           (ReviewVerifier)

planner.py       → agent_core.models (AgentMode) ONLY
suggester.py     → agent_core.models ONLY
patch_preview.py → agent_core.models ONLY
apply_mode.py    → agent_core.models ONLY
review_verifier.py → agent_core.models, agent_core.config
```

Döngüsel import riski: `planner.py`, `suggester.py`, `patch_preview.py`, `apply_mode.py` hiçbiri `orchestrator.py`'yi import etmez. `review_verifier.py` yalnızca `agent_core.config` ve `agent_core.models`'i import eder.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

Prework analizinden testable olarak işaretlenen kriterler:

- **1.4** (PROPERTY): Herhangi bir mock deps ile çalışırken Settings'e erişilmemeli
- **2.5** (PROPERTY): Herhangi bir geçersiz path/JSON için async_load boş KB döndürmeli
- **3.4** (PROPERTY): Herhangi bir stub policy enjekte edildiğinde orchestrator onu kullanmalı
- **4.3** (PROPERTY): Herhangi bir bilinmeyen mod için plan_label "Unknown" döndürmeli
- **6.1** (PROPERTY): Herhangi bir geçerli girdi için run() SessionRecord döndürmeli
- **6.2** (PROPERTY): Herhangi bir geçerli girdi için stream_run() AsyncIterable[dict] döndürmeli

**Redundancy analizi**:
- Property 6.1 ve 6.2 farklı metodları test eder; birleştirilemez.
- Property 1.4 ve 3.4 örtüşüyor gibi görünse de farklı şeyleri test eder: 1.4 Settings erişimini, 3.4 policy dispatch'ini test eder. Her ikisi de korunur.
- Property 2.5 tamamen bağımsız.
- Property 4.3 tamamen bağımsız.

Sonuç: 6 property, 2 tanesi (6.1 + 6.2) birleştirilebilir çünkü ikisi de "public API imzası korunur" temasını test eder. Birleştirilmiş hali daha kapsamlıdır.

---

### Property 1: Settings'e Bağımsız Çalışma

*For any* `OrchestratorDependencies` nesnesi (tüm alanları mock/stub ile doldurulmuş), `AgentOrchestrator(deps=mock_deps)` ile başlatılan orchestrator'ın `run()` çağrısı, `Settings` nesnesinin herhangi bir alanına doğrudan erişmeksizin tamamlanmalıdır.

**Validates: Requirements 1.4**

---

### Property 2: async_load Hata Toleransı

*For any* geçersiz dosya yolu (var olmayan path) veya geçersiz JSON içeriği (rastgele string), `KnowledgeBase.async_load(path)` çağrısı her zaman boş bir `KnowledgeBase` örneği döndürmeli ve bir `WARNING` logu üretmelidir; hiçbir zaman exception fırlatmamalıdır.

**Validates: Requirements 2.5**

---

### Property 3: Policy Enjeksiyonu

*For any* `Policy` protokolünü karşılayan stub nesne (yani `apply(parsed: ParsedAnswer) -> ParsedAnswer` metoduna sahip herhangi bir nesne), bu stub `OrchestratorDependencies.suggestion_policy` olarak geçirildiğinde, orchestrator'ın `SUGGEST` modunda çalıştırılması stub'ın `apply` metodunu çağırmalıdır.

**Validates: Requirements 3.4**

---

### Property 4: plan_label Bilinmeyen Mod Toleransı

*For any* `AgentMode` enum değerlerinden hiçbirine karşılık gelmeyen string veya değer, `AgentPlanner.plan_label()` çağrısı `ValueError` veya başka bir exception fırlatmaksızın `"Unknown"` string'ini döndürmelidir.

**Validates: Requirements 4.3**

---

### Property 5: Public API İmzası Korunması

*For any* geçerli `(mode, workspace_path, user_input)` kombinasyonu, refactor sonrasında `AgentOrchestrator.run()` çağrısı `SessionRecord` döndürmeli; `AgentOrchestrator.stream_run()` çağrısı `AsyncIterable[dict]` üretmelidir. Her iki metodun imzası ve dönüş tipleri değişmemelidir.

**Validates: Requirements 6.1, 6.2**

---

## Error Handling

### `KnowledgeBase.async_load()` Hataları

| Durum | Davranış |
|-------|----------|
| Dosya yok | Boş `KnowledgeBase()` döndür, `WARNING` logla |
| JSON geçersiz | Boş `KnowledgeBase()` döndür, `WARNING` logla |
| Dosya okuma izni yok | `OSError` yakalanır, boş KB döndür, `WARNING` logla |
| Pydantic validasyon hatası | Boş `KnowledgeBase()` döndür, `WARNING` logla |

### `AgentOrchestrator.__init__()` Geriye Dönük Uyumluluk

| Çağrı biçimi | Davranış |
|--------------|----------|
| `AgentOrchestrator(settings)` | Senkron `_build_deps_sync(settings)` çağrılır |
| `AgentOrchestrator(deps=mock_deps)` | `mock_deps` doğrudan kullanılır |
| `AgentOrchestrator(deps_obj)` | `deps_obj` doğrudan kullanılır |
| `AgentOrchestrator()` | `TypeError` fırlatılır |

### `AgentPlanner.plan_label()` Bilinmeyen Mod

Mevcut davranış: `"Suggest"` döndürür (default). Yeni davranış: `"Unknown"` döndürür. Bu değişiklik geriye dönük uyumlu değildir ancak requirements 4.3 açıkça `"Unknown"` döndürülmesini zorunlu kılar. `plan_label` sonucu yalnızca loglama/etiketleme amacıyla kullanıldığından bu değişiklik güvenlidir.

---

## Testing Strategy

Bu refactor, saf fonksiyon mantığı (policy uygulaması, planner etiketleme) ve bağımlılık enjeksiyonu içerdiğinden property-based testing uygundur.

**PBT Kütüphanesi**: `hypothesis` (Python ekosisteminin standart PBT kütüphanesi)

### Birim Testleri

Aşağıdaki senaryolar için örnek tabanlı testler yazılır:

- `OrchestratorDependencies` tüm alanlarla başarıyla oluşturulabilir
- `AgentOrchestrator.from_settings(settings)` doğru tip döndürür
- `AgentOrchestrator(settings)` (eski kullanım) `TypeError` fırlatmaz
- `KnowledgeBase.async_load()` var olan geçerli dosyayı doğru yükler
- `KnowledgeBase.load()` hâlâ çalışır (geriye dönük uyumluluk)
- `Policy` protokolü `isinstance` ile doğrulanabilir
- `AgentPlanner.plan_steps()` boş liste döndürür
- `agent_core.agent.__init__` yalnızca `AgentOrchestrator` ve `OrchestratorDependencies` export eder
- `from agent_core.agent import AgentOrchestrator` döngüsel import olmadan çalışır

### Property-Based Testler

Her property testi minimum 100 iterasyon çalıştırılır. Her test, ilgili tasarım property'sine referans içerir.

**Feature: orchestrator-refactor, Property 1: Settings'e Bağımsız Çalışma**
- Generator: Tüm alanları `MagicMock()` ile doldurulmuş `OrchestratorDependencies` nesneleri
- Assertion: `run()` tamamlandığında `Settings` nesnesi erişilmemiş olmalı

**Feature: orchestrator-refactor, Property 2: async_load Hata Toleransı**
- Generator: `hypothesis.strategies.text()` ile rastgele geçersiz JSON string'leri; `hypothesis.strategies.builds(Path, text())` ile rastgele geçersiz path'ler
- Assertion: Her zaman `KnowledgeBase` örneği döner, exception fırlatılmaz

**Feature: orchestrator-refactor, Property 3: Policy Enjeksiyonu**
- Generator: `apply(parsed) -> parsed` imzasını karşılayan rastgele stub nesneler (lambda veya MagicMock)
- Assertion: `SUGGEST` modunda `run()` çağrısı stub'ın `apply` metodunu çağırır

**Feature: orchestrator-refactor, Property 4: plan_label Bilinmeyen Mod Toleransı**
- Generator: `hypothesis.strategies.text()` ile rastgele string'ler (geçerli AgentMode değerleri hariç)
- Assertion: Her zaman `"Unknown"` döner, exception fırlatılmaz

**Feature: orchestrator-refactor, Property 5: Public API İmzası Korunması**
- Generator: Geçerli `AgentMode` değerleri, geçici `tmp_path` dizinleri, rastgele `user_input` string'leri
- Assertion: `run()` her zaman `SessionRecord` döndürür; `stream_run()` her zaman async iterable üretir

### Regresyon Testleri

`tests/test_agent_core.py` dosyası değiştirilmeden tüm testlerin geçmesi zorunludur. Bu, refactor'ın geriye dönük uyumluluğunun temel kanıtıdır.

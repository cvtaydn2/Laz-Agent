# Requirements Document

## Introduction

Bu spec, `editor-agent` projesindeki `AgentOrchestrator` sınıfının mimari refactor'ını kapsar. Mevcut yapıda orchestrator doğrudan 14+ bağımlılığı `__init__` içinde somutlaştırmakta, policy sınıfları dışarıdan enjekte edilememekte ve `KnowledgeBase.load()` senkron olarak çağrılmaktadır. Bu refactor; test edilebilirliği artırmak, bağımlılık enjeksiyonunu standartlaştırmak ve async/sync sınırını netleştirmek amacıyla yapılmaktadır. Mevcut davranış ve public API korunacak; yalnızca iç yapı yeniden düzenlenecektir.

## Glossary

- **Orchestrator**: `AgentOrchestrator` sınıfı — agent çalışma döngüsünü koordine eden ana bileşen.
- **OrchestratorDependencies**: Orchestrator'ın ihtiyaç duyduğu tüm bağımlılıkları bir arada tutan veri nesnesi (dataclass veya Pydantic modeli).
- **Policy**: `SuggestionPolicy`, `PatchPreviewPolicy`, `ApplyModePolicy` gibi `ParsedAnswer` üzerinde dönüşüm uygulayan sınıflar.
- **WorkspaceServices**: `WorkspaceScanner`, `WorkspaceRanker`, `WorkspaceReader` üçlüsünü temsil eden kavram.
- **OutputWriters**: `SessionWriter`, `PatchProposalWriter`, `ApplyLogWriter` üçlüsünü temsil eden kavram.
- **KnowledgeBase**: Geçmiş başarılı düzeltmeleri saklayan ve sorgulayan bileşen.
- **LLMProvider**: `NvidiaProvider` gibi LLM arka uçlarını soyutlayan arayüz.
- **Settings**: Uygulama yapılandırmasını tutan Pydantic modeli.
- **AsyncIO**: Python'un yerleşik asenkron I/O çerçevesi.
- **DI (Dependency Injection)**: Bağımlılıkların dışarıdan verilmesi deseni.
- **God Object**: Çok fazla sorumluluğu tek bir sınıfta toplayan anti-pattern.

---

## Requirements

### Requirement 1: Bağımlılık Enjeksiyonu ile Orchestrator Başlatma

**User Story:** Bir geliştirici olarak, `AgentOrchestrator`'ı test ederken gerçek dosya sistemi veya ağ çağrısı yapmadan tüm bağımlılıkları mock'layabilmek istiyorum; böylece birim testleri hızlı ve güvenilir çalışır.

#### Acceptance Criteria

1. THE `Orchestrator` SHALL `OrchestratorDependencies` nesnesini `__init__` parametresi olarak kabul etmek.
2. WHEN `OrchestratorDependencies` sağlanmadığında, THE `Orchestrator` SHALL `Settings` nesnesinden varsayılan bağımlılıkları oluşturan bir fabrika metodu (`from_settings`) aracılığıyla başlatılabilmek.
3. THE `OrchestratorDependencies` SHALL en az şu alanları içermek: `scanner`, `ranker`, `reader`, `llm_provider`, `planner`, `response_parser`, `suggestion_policy`, `patch_preview_policy`, `apply_policy`, `review_verifier`, `session_writer`, `patch_writer`, `apply_log_writer`, `apply_engine`, `knowledge_base`.
4. WHEN bir test `AgentOrchestrator(deps=mock_deps)` şeklinde başlatıldığında, THE `Orchestrator` SHALL `Settings` nesnesine doğrudan erişmeksizin çalışmak.
5. THE `from_settings` fabrika metodu SHALL mevcut `AgentOrchestrator(settings)` çağrılarıyla geriye dönük uyumlu olmak.

---

### Requirement 2: Senkron KnowledgeBase Yüklemesinin Async'e Taşınması

**User Story:** Bir geliştirici olarak, `KnowledgeBase.load()` çağrısının async event loop'u bloke etmemesini istiyorum; böylece yüksek eşzamanlılık senaryolarında gecikme yaşanmaz.

#### Acceptance Criteria

1. THE `KnowledgeBase` SHALL `async_load(path: Path) -> KnowledgeBase` adında bir async sınıf metodu sunmak.
2. WHEN `async_load` çağrıldığında, THE `KnowledgeBase` SHALL dosya okuma işlemini `asyncio.to_thread` veya eşdeğer bir mekanizma ile event loop dışında yürütmek.
3. THE `from_settings` fabrika metodu SHALL `KnowledgeBase.async_load()` metodunu `await` ile çağırmak; senkron `KnowledgeBase.load()` çağrısını `__init__` içinde kullanmamak.
4. THE mevcut senkron `KnowledgeBase.load()` metodu SHALL geriye dönük uyumluluk için korunmak; yalnızca async olmayan bağlamlarda kullanılmak üzere işaretlenmek.
5. IF `async_load` sırasında dosya okunamaz veya JSON geçersizse, THEN THE `KnowledgeBase` SHALL boş bir `KnowledgeBase` örneği döndürmek ve hatayı `WARNING` seviyesinde loglamak.

---

### Requirement 3: Policy Sınıflarının Protokol Arayüzü ile Soyutlanması

**User Story:** Bir geliştirici olarak, `SuggestionPolicy`, `PatchPreviewPolicy` ve `ApplyModePolicy` sınıflarını test sırasında özel implementasyonlarla değiştirebilmek istiyorum; böylece orchestrator mantığını policy davranışından bağımsız test edebilirim.

#### Acceptance Criteria

1. THE `Orchestrator` SHALL `Policy` protokolünü tanımlamak: `def apply(self, parsed: ParsedAnswer) -> ParsedAnswer`.
2. THE `SuggestionPolicy`, `PatchPreviewPolicy` ve `ApplyModePolicy` sınıfları SHALL `Policy` protokolünü karşılamak (structural subtyping — `Protocol` sınıfından miras almak zorunda değil).
3. THE `OrchestratorDependencies` SHALL `suggestion_policy`, `patch_preview_policy` ve `apply_policy` alanlarını `Policy` protokol tipiyle tanımlamak.
4. WHEN bir test `OrchestratorDependencies` içine lambda veya stub policy geçirdiğinde, THE `Orchestrator` SHALL bu policy'yi gerçek implementasyon yerine kullanmak.
5. THE `ReviewVerifier` SHALL `verify(workspace_path, parsed, selected_context) -> ParsedAnswer` imzasını koruyan bir protokol arayüzüyle soyutlanmak.

---

### Requirement 4: AgentPlanner İçeriğinin Netleştirilmesi ve Workspace Taramasına Dahil Edilmesi

**User Story:** Bir geliştirici olarak, `AgentPlanner`'ın ne yaptığını ve neden workspace taramasına dahil edilmediğini anlamak istiyorum; böylece gelecekteki genişletmeler için doğru yere kod ekleyebilirim.

#### Acceptance Criteria

1. THE `AgentPlanner` SHALL modül düzeyinde bir docstring içermek; sınıfın sorumluluğunu ve `plan_label` metodunun amacını açıklamak.
2. THE `WorkspaceScanner` SHALL `planner.py` dosyasını tarama kapsamına dahil etmek (mevcut `agent/` dizini tarama dışı bırakılıyorsa bu kural geçersizdir; tarama kuralları `Settings.DEFAULT_ALLOWED_EXTENSIONS` ile belirlenir).
3. WHEN `AgentPlanner.plan_label` bilinmeyen bir `AgentMode` değeri aldığında, THE `AgentPlanner` SHALL `ValueError` fırlatmak yerine `"Unknown"` döndürmek.
4. THE `AgentPlanner` SHALL gelecekteki genişletmeler için `plan_steps(mode: AgentMode) -> list[str]` metodunu iskelet olarak içermek; şimdilik boş liste döndürmek.

---

### Requirement 5: Döngüsel Import Riskinin Giderilmesi

**User Story:** Bir geliştirici olarak, `agent_core/agent/` paketi içindeki modüllerin birbirini döngüsel olarak import etmediğinden emin olmak istiyorum; böylece `ImportError` riski olmadan yeni modüller ekleyebilirim.

#### Acceptance Criteria

1. THE `agent_core.agent` paketi SHALL `orchestrator.py` dışındaki modüllerin `orchestrator.py`'yi import etmemesini sağlamak.
2. THE `agent_core.agent.__init__` modülü SHALL yalnızca `AgentOrchestrator` ve `OrchestratorDependencies` sınıflarını dışa aktarmak; diğer iç modülleri doğrudan expose etmemek.
3. WHEN `python -c "from agent_core.agent import AgentOrchestrator"` komutu çalıştırıldığında, THE Python yorumlayıcısı SHALL `ImportError` veya `CircularImportError` fırlatmamak.
4. THE `OrchestratorDependencies` veri nesnesi SHALL `orchestrator.py` içinde veya ayrı bir `deps.py` modülünde tanımlanmak; policy modüllerinden import edilmemek.

---

### Requirement 6: Mevcut Public API ve Davranışın Korunması

**User Story:** Bir geliştirici olarak, refactor sonrasında mevcut `server/service.py`, `main.py` ve test dosyalarının değiştirilmeden çalışmaya devam etmesini istiyorum; böylece refactor sırasında regresyon riski minimize edilir.

#### Acceptance Criteria

1. THE `AgentOrchestrator` SHALL `run(mode, workspace_path, user_input, ...)` metodunu aynı imza ve dönüş tipiyle (`SessionRecord`) korumak.
2. THE `AgentOrchestrator` SHALL `stream_run(...)` metodunu aynı imza ve dönüş tipiyle (`AsyncIterable[dict]`) korumak.
3. THE `AgentOrchestrator` SHALL `rollback(session_id)` ve `self_heal(workspace_path, max_retries)` metodlarını korumak.
4. WHEN `AgentOrchestrator(settings)` şeklinde başlatıldığında (mevcut kullanım), THE `Orchestrator` SHALL `TypeError` fırlatmamak ve normal şekilde çalışmak.
5. THE mevcut test dosyası `tests/test_agent_core.py` SHALL refactor sonrasında herhangi bir değişiklik yapılmadan tüm testleri geçmek.
6. IF `AgentOrchestrator` `OrchestratorDependencies` ile başlatıldığında `settings` alanına ihtiyaç duyulursa, THEN THE `OrchestratorDependencies` SHALL opsiyonel bir `settings: Settings | None` alanı içermek.

---

### Requirement 7: Refactor Kapsamı Dışında Kalan Bileşenler

**User Story:** Bir geliştirici olarak, bu refactor'ın hangi bileşenlere dokunmadığını bilmek istiyorum; böylece kapsam kayması (scope creep) yaşanmaz.

#### Acceptance Criteria

1. THE refactor SHALL `WorkspaceScanner`, `WorkspaceRanker`, `WorkspaceReader` sınıflarının iç implementasyonunu değiştirmemek.
2. THE refactor SHALL `LLMProvider` arayüzünü ve `NvidiaProvider` implementasyonunu değiştirmemek.
3. THE refactor SHALL `agent_core/models.py`, `agent_core/prompts.py` ve `agent_core/server/` paketini değiştirmemek.
4. THE refactor SHALL `ApplyEngine`, `SessionWriter`, `PatchProposalWriter`, `ApplyLogWriter` sınıflarının iç implementasyonunu değiştirmemek.
5. THE refactor SHALL `ResponseParser` sınıfının parse mantığını değiştirmemek; yalnızca bağımlılık enjeksiyonu için arayüzini güncellemek.

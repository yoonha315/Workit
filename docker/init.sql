-- ================================================================
-- Workit 테이블 DDL + 초기 데이터
-- DB: PostgreSQL 15
-- 기준: 테이블_정의서.xlsx
-- ================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ----------------------------------------------------------------
-- 1. users (사용자)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    name            VARCHAR(50)     NOT NULL,
    email           VARCHAR(100)    NOT NULL,
    login_id        VARCHAR(100)    NOT NULL,
    password        VARCHAR(255)    NOT NULL,
    phone           VARCHAR(50),
    department      VARCHAR(50),
    organization    VARCHAR(100),
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_users          PRIMARY KEY (id),
    CONSTRAINT uq_users_email    UNIQUE (email),
    CONSTRAINT uq_users_login_id UNIQUE (login_id)
);

COMMENT ON TABLE  users              IS '교육청 담당 공무원';
COMMENT ON COLUMN users.login_id     IS '로그인 ID';
COMMENT ON COLUMN users.department   IS '예: 정보화사업팀';
COMMENT ON COLUMN users.organization IS '예: 서울특별시교육청';


-- ----------------------------------------------------------------
-- 2. companies (SI 기업)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    company_name    VARCHAR(100)    NOT NULL,
    business_no     VARCHAR(20),
    phone           VARCHAR(50),
    email           VARCHAR(100),
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_companies PRIMARY KEY (id)
);

COMMENT ON TABLE companies IS 'SI 기업 (계약 상대방)';


-- ----------------------------------------------------------------
-- 3. contracts (계약서 목록)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS contracts (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    company_id      UUID            NOT NULL,
    uploaded_by     UUID            NOT NULL,
    title           VARCHAR(200)    NOT NULL,
    file_path       VARCHAR(500)    NOT NULL,
    status          VARCHAR(20)     NOT NULL DEFAULT '검토중',
    start_date      DATE,
    end_date        DATE,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_contracts         PRIMARY KEY (id),
    CONSTRAINT fk_contracts_company FOREIGN KEY (company_id)  REFERENCES companies (id),
    CONSTRAINT fk_contracts_user    FOREIGN KEY (uploaded_by) REFERENCES users (id),
    CONSTRAINT ck_contracts_status  CHECK (status IN ('검토중','완료','이행중','만료'))
);

COMMENT ON TABLE  contracts           IS '계약서 목록';
COMMENT ON COLUMN contracts.file_path IS 'S3 저장 경로';
COMMENT ON COLUMN contracts.status    IS '검토중 / 완료 / 이행중 / 만료';


-- ----------------------------------------------------------------
-- 4. deliverables (이행관리 산출물)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deliverables (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    contract_id     UUID            NOT NULL,
    company_id      UUID            NOT NULL,
    uploaded_by     UUID            NOT NULL,
    doc_type        VARCHAR(30)     NOT NULL,
    title           VARCHAR(200)    NOT NULL,
    file_path       VARCHAR(500)    NOT NULL,
    status          VARCHAR(20)     NOT NULL DEFAULT '검토중',
    due_date        DATE,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_deliverables          PRIMARY KEY (id),
    CONSTRAINT fk_deliverables_contract FOREIGN KEY (contract_id) REFERENCES contracts (id),
    CONSTRAINT fk_deliverables_company  FOREIGN KEY (company_id)  REFERENCES companies (id),
    CONSTRAINT fk_deliverables_user     FOREIGN KEY (uploaded_by) REFERENCES users (id),
    CONSTRAINT ck_deliverables_doc_type CHECK (doc_type IN ('수행계획서','테스트계획서','테스트결과보고서','결과보고서')),
    CONSTRAINT ck_deliverables_status   CHECK (status IN ('검토중','보완요청됨','완료'))
);

COMMENT ON TABLE  deliverables           IS '이행관리 산출물';
COMMENT ON COLUMN deliverables.doc_type  IS '수행계획서 / 테스트계획서 / 테스트결과보고서 / 결과보고서';
COMMENT ON COLUMN deliverables.status    IS '검토중 / 보완요청됨 / 완료';
COMMENT ON COLUMN deliverables.file_path IS 'S3 저장 경로';


-- ----------------------------------------------------------------
-- 5. review_results (AI 피드백 결과)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_results (
    id                  UUID            NOT NULL DEFAULT uuid_generate_v4(),
    deliverable_id      UUID,
    contract_id         UUID,
    review_type         VARCHAR(20)     NOT NULL,
    requirement_id      VARCHAR(50),
    check_type          VARCHAR(30)     NOT NULL,
    result              VARCHAR(20)     NOT NULL,
    feedback            TEXT,
    detected_at         TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_review_results     PRIMARY KEY (id),
    CONSTRAINT fk_review_deliverable FOREIGN KEY (deliverable_id) REFERENCES deliverables (id),
    CONSTRAINT fk_review_contract    FOREIGN KEY (contract_id)    REFERENCES contracts (id),
    CONSTRAINT ck_review_type        CHECK (review_type IN ('contract','deliverable')),
    CONSTRAINT ck_review_check_type  CHECK (check_type IN ('누락','정합성오류','품질미달','위험조항')),
    CONSTRAINT ck_review_result      CHECK (result IN ('통과','미흡','누락'))
);

COMMENT ON TABLE  review_results             IS 'AI 피드백 결과';
COMMENT ON COLUMN review_results.review_type IS 'contract: 계약서 / deliverable: 산출물';


-- ----------------------------------------------------------------
-- 6. audit_logs (보안 감사 로그)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id              UUID            NOT NULL DEFAULT uuid_generate_v4(),
    user_id         UUID            NOT NULL,
    action          VARCHAR(30)     NOT NULL,
    target_type     VARCHAR(30),
    target_id       UUID,
    ip_address      VARCHAR(45),
    performed_at    TIMESTAMP       NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_audit_logs        PRIMARY KEY (id),
    CONSTRAINT fk_audit_user        FOREIGN KEY (user_id) REFERENCES users (id),
    CONSTRAINT ck_audit_action      CHECK (action IN ('login','logout','upload','analyze','download','request_revision')),
    CONSTRAINT ck_audit_target_type CHECK (target_type IN ('contract','deliverable','review_result') OR target_type IS NULL)
);

COMMENT ON TABLE  audit_logs            IS '보안 감사 로그 (6개월 이상 보관)';
COMMENT ON COLUMN audit_logs.ip_address IS '접속 IP (IPv6 포함, 최대 45자)';


-- ----------------------------------------------------------------
-- 7. knowledge_logs (지식 로그)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_logs (
    id                  UUID        NOT NULL DEFAULT uuid_generate_v4(),
    contract_id         UUID,
    deliverable_id      UUID,
    reviewed_by         UUID,
    doc_type            VARCHAR(30),
    error_pattern       VARCHAR(100),
    feedback_summary    TEXT,
    created_at          TIMESTAMP   NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_knowledge_logs        PRIMARY KEY (id),
    CONSTRAINT fk_knowledge_contract    FOREIGN KEY (contract_id)    REFERENCES contracts (id),
    CONSTRAINT fk_knowledge_deliverable FOREIGN KEY (deliverable_id) REFERENCES deliverables (id),
    CONSTRAINT fk_knowledge_user        FOREIGN KEY (reviewed_by)    REFERENCES users (id)
);

COMMENT ON TABLE  knowledge_logs                  IS '지식 로그';
COMMENT ON COLUMN knowledge_logs.feedback_summary IS 'Qdrant review_knowledge 컬렉션 연동';


-- ================================================================
-- 인덱스
-- ================================================================
CREATE INDEX IF NOT EXISTS idx_contracts_company      ON contracts      (company_id);
CREATE INDEX IF NOT EXISTS idx_contracts_uploaded_by  ON contracts      (uploaded_by);
CREATE INDEX IF NOT EXISTS idx_contracts_status       ON contracts      (status);

CREATE INDEX IF NOT EXISTS idx_deliverables_contract  ON deliverables   (contract_id);
CREATE INDEX IF NOT EXISTS idx_deliverables_company   ON deliverables   (company_id);
CREATE INDEX IF NOT EXISTS idx_deliverables_status    ON deliverables   (status);
CREATE INDEX IF NOT EXISTS idx_deliverables_doc_type  ON deliverables   (doc_type);

CREATE INDEX IF NOT EXISTS idx_review_deliverable     ON review_results (deliverable_id);
CREATE INDEX IF NOT EXISTS idx_review_contract        ON review_results (contract_id);
CREATE INDEX IF NOT EXISTS idx_review_type            ON review_results (review_type);

CREATE INDEX IF NOT EXISTS idx_audit_user             ON audit_logs     (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_performed_at     ON audit_logs     (performed_at);

CREATE INDEX IF NOT EXISTS idx_knowledge_contract     ON knowledge_logs (contract_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_deliverable  ON knowledge_logs (deliverable_id);


-- ================================================================
-- 초기 데이터 INSERT
-- ================================================================

-- 1. users
INSERT INTO users (name, email, login_id, password, phone, department, organization)
VALUES (
    '김담당',
    'kim.damdam@sen.go.kr',
    'kim.damdam',
    '$2b$12$examplehashedpassword1234567890',
    '02-1234-5678',
    '정보화사업팀',
    '서울특별시교육청'
) ON CONFLICT (login_id) DO NOTHING;


-- 2. companies
INSERT INTO companies (company_name, business_no, phone, email)
VALUES (
    'A소프트웨어',
    '123-45-67890',
    '02-9876-5432',
    'contact@asoft.co.kr'
) ON CONFLICT DO NOTHING;


-- 3. contracts
INSERT INTO contracts (company_id, uploaded_by, title, file_path, status, start_date, end_date)
SELECT
    c.id,
    u.id,
    '2025년 교육행정정보시스템 고도화 사업 계약서',
    'contracts/2025-edu-admin/contract.pdf',
    '이행중',
    '2025-01-01',
    '2025-12-31'
FROM companies c, users u
WHERE c.company_name = 'A소프트웨어'
  AND u.login_id     = 'kim.damdam'
ON CONFLICT DO NOTHING;


-- 4. deliverables
INSERT INTO deliverables (contract_id, company_id, uploaded_by, doc_type, title, file_path, status, due_date)
SELECT
    ct.id,
    c.id,
    u.id,
    '수행계획서',
    '2025년 교육행정정보시스템 고도화 사업 수행계획서 v1.0',
    'deliverables/2025-edu-admin/수행계획서_v1.0.pdf',
    '검토중',
    '2025-02-01'
FROM contracts ct
JOIN companies c ON c.company_name = 'A소프트웨어'
JOIN users    u ON u.login_id      = 'kim.damdam'
WHERE ct.title = '2025년 교육행정정보시스템 고도화 사업 계약서'
ON CONFLICT DO NOTHING;


-- 5. review_results
INSERT INTO review_results (deliverable_id, contract_id, review_type, requirement_id, check_type, result, feedback)
SELECT
    d.id,
    NULL,
    'deliverable',
    'REQ-013',
    '누락',
    '미흡',
    '수행계획서 3.2절에 WBS가 누락되어 있습니다. 전자정부 SW사업 산출물 관리 가이드 기준에 따라 사업 일정 및 인력 투입 계획이 포함된 WBS를 작성하여 보완해주세요.'
FROM deliverables d
WHERE d.title = '2025년 교육행정정보시스템 고도화 사업 수행계획서 v1.0'
ON CONFLICT DO NOTHING;


-- 6. audit_logs
INSERT INTO audit_logs (user_id, action, target_type, target_id, ip_address)
SELECT
    u.id,
    'upload',
    'deliverable',
    d.id,
    '203.249.121.45'
FROM users u, deliverables d
WHERE u.login_id = 'kim.damdam'
  AND d.title    = '2025년 교육행정정보시스템 고도화 사업 수행계획서 v1.0'
ON CONFLICT DO NOTHING;


-- 7. knowledge_logs
INSERT INTO knowledge_logs (contract_id, deliverable_id, reviewed_by, doc_type, error_pattern, feedback_summary)
SELECT
    ct.id,
    d.id,
    u.id,
    '수행계획서',
    'WBS누락',
    '수행계획서에서 WBS 누락이 반복적으로 발생. 3.2절 일정 계획 섹션 집중 검토 필요.'
FROM contracts  ct
JOIN deliverables d  ON d.title    = '2025년 교육행정정보시스템 고도화 사업 수행계획서 v1.0'
JOIN users        u  ON u.login_id = 'kim.damdam'
WHERE ct.title = '2025년 교육행정정보시스템 고도화 사업 계약서'
ON CONFLICT DO NOTHING;

-- 1. users
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('김민재', 'jeonghoan@example.com', 'user001', '$2b$12$f64100c6559a17250f3d3f7d58649040', '061-594-1871', '정보화사업팀', '서울특별시교육청', '2025-04-08 10:06:59');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('김성진', 'hgim@example.org', 'user002', '$2b$12$c935a3b8cba0e5cb6499beb8cd28e3f1', '061-244-9019', '시스템운영팀', '경기도교육청', '2026-01-22 03:02:55');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('심동현', 'seoyeongson@example.com', 'user003', '$2b$12$74c4ccd8111ee87a476b336c3b1f8ae5', '010-0961-1967', '디지털혁신팀', '경기도교육청', '2025-01-28 17:31:34');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('이상호', 'sgim@example.org', 'user004', '$2b$12$4a906d3f13391ae4c3e77ffce7d7a4e4', '052-289-6811', '정보화사업팀', '대전광역시교육청', '2026-01-25 00:40:22');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('강병철', 'mijeongan@example.net', 'user005', '$2b$12$cbc735a6468a5a9476caccc38301a45b', '054-158-0128', '정보화사업팀', '대전광역시교육청', '2026-05-26 17:56:47');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('김옥순', 'jaehyeonbaeg@example.net', 'user006', '$2b$12$7bdc7cdd914df6bbe63e28420599ed00', '054-953-6745', '정보보안팀', '서울특별시교육청', '2025-06-18 18:18:38');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('김하윤', 'gyeongsuca@example.com', 'user007', '$2b$12$5f465f4894046b17fbf20a73629937e9', '032-493-0474', '정보화사업팀', '서울특별시교육청', '2025-01-15 01:40:00');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('이정웅', 'xgim@example.org', 'user008', '$2b$12$825e602b8905125e64f44d581ec8fcc4', '064-702-1907', '디지털혁신팀', '경기도교육청', '2025-08-08 01:00:30');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('서성현', 'yeongilgim@example.net', 'user009', '$2b$12$a2c2490143ca5392464d6dce79f2f298', '032-582-8076', '행정지원팀', '대전광역시교육청', '2025-08-28 13:55:57');
INSERT INTO users (name, email, login_id, password, phone, department, organization, created_at) VALUES ('이준서', 'tu@example.com', 'user010', '$2b$12$0f5d4f07fcada5cb09c2e200b23fdedc', '070-0903-8354', '정보화사업팀', '대전광역시교육청', '2025-10-06 19:28:42');

-- 2. companies
INSERT INTO companies (company_name, business_no, phone, email, created_at) VALUES ('(주)테크솔루션', '303-93-81426', '051-778-2267', 'contact@company1.co.kr', '2024-03-17 06:43:53');
INSERT INTO companies (company_name, business_no, phone, email, created_at) VALUES ('(주)디지털파트너스', '529-38-68878', '02-6463-2845', 'contact@company2.co.kr', '2024-05-17 10:06:25');
INSERT INTO companies (company_name, business_no, phone, email, created_at) VALUES ('스마트시스템즈(주)', '703-45-10851', '018-778-7809', 'contact@company3.co.kr', '2024-05-23 18:16:55');
INSERT INTO companies (company_name, business_no, phone, email, created_at) VALUES ('(주)이노베이션IT', '877-30-65392', '064-703-2692', 'contact@company4.co.kr', '2025-02-17 15:43:57');
INSERT INTO companies (company_name, business_no, phone, email, created_at) VALUES ('클라우드웍스(주)', '448-45-30379', '054-728-9129', 'contact@company5.co.kr', '2024-12-14 07:54:50');
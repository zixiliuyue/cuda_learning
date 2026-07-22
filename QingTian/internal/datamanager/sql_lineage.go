package datamanager

// DatasetLineageSchema defines the SQL schema for dataset lineage tracking.
//
// CREATE TABLE dataset_lineage (
//	    id BIGINT PRIMARY KEY AUTO_INCREMENT,
//	    dataset_id VARCHAR(64) NOT NULL,
//	    version_id VARCHAR(64) NOT NULL,
//	    consumed_by_task_id VARCHAR(64) NOT NULL,
//	    consumed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//	    produced_model_id VARCHAR(64),
//	    INDEX (dataset_id),
//	    INDEX (consumed_by_task_id)
//	);
const DatasetLineageSchema = `
CREATE TABLE IF NOT EXISTS dataset_lineage (
	id BIGINT PRIMARY KEY AUTO_INCREMENT,
	dataset_id VARCHAR(64) NOT NULL,
	version_id VARCHAR(64) NOT NULL,
	consumed_by_task_id VARCHAR(64) NOT NULL,
	consumed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	produced_model_id VARCHAR(64),
	INDEX (dataset_id),
	INDEX (consumed_by_task_id)
);
`

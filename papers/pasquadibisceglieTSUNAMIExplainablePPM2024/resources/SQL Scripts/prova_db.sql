DROP DATABASE IF EXISTS `prova_db`;
CREATE DATABASE `prova_db`;
USE `prova_db`;
DROP TABLE IF EXISTS `receipts`;

CREATE TABLE `receipts` (
  `K_Receipt` varchar(100) NOT NULL,
  `K_Member` varchar(100) NOT NULL,
  `Quantity` int DEFAULT NULL,
  `Q_Amount` decimal(20,2) DEFAULT NULL,
  `Q_Discount_Amount` int DEFAULT NULL,
  `T_Receipt` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`K_Receipt`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

LOCK TABLES `receipts` WRITE;
/*!40000 ALTER TABLE `receipts` DISABLE KEYS */;
INSERT INTO `receipts` VALUES
('t0','paul',2,4,NULL,'2023-01-01 13:45:00'),
('t0_1','andrea',1,3,NULL,'2023-01-02 13:45:00'),
('t1','paul',1,1,NULL,'2023-01-03 10:10:00'),
('t1_1','giusy',1,3,NULL,'2023-01-04 13:45:00'),
('t2','paul',1,3,NULL,'2023-01-05 11:10:00'),
('t2_1','gianluca',1,3,NULL,'2023-01-06 13:45:00'),
('t3_1','vincenzo',1,3,NULL,'2023-01-07 13:45:00'),
('t3','paul',1,1,NULL,'2023-01-08 10:00:00'),
('t4','paul',1,1,NULL,'2023-01-08 11:00:00'),
('t5','paul',1,3,NULL,'2023-01-08 12:00:00'),
('t6','pietro',1,3,NULL,'2023-01-09 12:00:00'),
('t7','sara',1,3,NULL,'2023-01-10 12:00:00'),
('t8','vito',1,3,NULL,'2023-01-11 12:00:00'),
('t9','vito',1,3,NULL,'2023-01-12 12:00:00'),
('t10','vito',1,3,NULL,'2023-01-13 12:00:00'),
('t11','vito',1,3,NULL,'2023-01-14 12:00:00');



UNLOCK TABLES;
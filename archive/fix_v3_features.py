"""修复V3分类器的特征维度问题"""
import sys
import os
import io

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取原文件
with open('src/skill_extraction/v3_skill_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换prepare_training_data方法中的代码
old_code = """        # 正样本：种子词典
        logger.info("提取正样本（技能词）...")
        for word in self.skill_seeds.keys():
            features = self.extract_features(word)
            X.append(features)
            y.append(1)  # 1 = 是技能
            words.append(word)
        
        logger.info(f"正样本数量: {len([label for label in y if label == 1])}")
        
        # 负样本：黑名单
        logger.info("提取负样本（非技能词）...")
        all_blacklist_words = self.dict_manager.get_all_blacklist_words()
        for word in all_blacklist_words:
            features = self.extract_features(word)
            X.append(features)
            y.append(0)  # 0 = 不是技能
            words.append(word)
        
        logger.info(f"负样本数量: {len([label for label in y if label == 0])}")
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)"""

new_code = """        # 正样本：种子词典
        logger.info("提取正样本（技能词）...")
        for word in self.skill_seeds.keys():
            features = self.extract_features(word)
            # 确保特征是125维
            if len(features) != 125:
                logger.warning(f"词 '{word}' 特征维度异常: {len(features)}, 跳过")
                continue
            X.append(features.tolist())
            y.append(1)  # 1 = 是技能
            words.append(word)
        
        logger.info(f"正样本数量: {len([label for label in y if label == 1])}")
        
        # 负样本：黑名单
        logger.info("提取负样本（非技能词）...")
        all_blacklist_words = self.dict_manager.get_all_blacklist_words()
        for word in all_blacklist_words:
            features = self.extract_features(word)
            # 确保特征是125维
            if len(features) != 125:
                logger.warning(f"词 '{word}' 特征维度异常: {len(features)}, 跳过")
                continue
            X.append(features.tolist())
            y.append(0)  # 0 = 不是技能
            words.append(word)
        
        logger.info(f"负样本数量: {len([label for label in y if label == 0])}")
        
        # 转换为numpy数组
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int32)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("OK: Fixed prepare_training_data method")
else:
    print("ERROR: Target code not found")
    sys.exit(1)

# 写回文件
with open('src/skill_extraction/v3_skill_classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("OK: File updated")

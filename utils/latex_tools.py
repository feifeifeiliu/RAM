def format_vald_metrics(mean_dict):
    """
    将mean_dict中vald相关的内容格式化为LaTeX表格格式
    顺序：FID, R1, R2, R3, Matching Score, Diversity
    pm后的数字去掉前面的0
    """
    def format_metric(value, conf, decimals=3):
        """格式化单个指标，pm后的数字去掉前面的0"""
        conf_str = f"{conf:.{decimals}f}"
        # 如果conf以"0."开头，去掉"0"
        if conf_str.startswith("0."):
            conf_str = conf_str[1:]  # 去掉开头的"0"
        return f"${value:.{decimals}f}^{{\\pm {conf_str}}}$"
    
    metrics = []
    
    # FID
    fid_val = mean_dict['FID_vald']
    fid_conf = mean_dict['FID_conf_vald']
    metrics.append(format_metric(fid_val, fid_conf))
    
    # R-Precision (R1, R2, R3)
    r_precision_vals = mean_dict['R_precision_vald']
    r_precision_confs = mean_dict['R_precision_conf_vald']
    for i in range(3):
        metrics.append(format_metric(r_precision_vals[i], r_precision_confs[i]))
    
    # Matching Score
    ms_val = mean_dict['Matching Score_vald']
    ms_conf = mean_dict['Matching Score_conf_vald']
    metrics.append(format_metric(ms_val, ms_conf))
    
    # Diversity
    div_val = mean_dict['Diversity_vald']
    div_conf = mean_dict['Diversity_conf_vald']
    metrics.append(format_metric(div_val, div_conf))
    
    # 输出格式化结果
    result = " & ".join(metrics)
    return "& " + result
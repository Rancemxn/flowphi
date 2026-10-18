"""Facts recovered from the supplied Android binary; separate from planner tolerances."""

JUDGEMENT_SUMMARY = {
    '来源': '给定 Phigros APK 的静态默认值；特殊模式/运行时值可能不同，未经设备实测。',
    'Tap': '需要新按下。时间差绝对值 <80ms 为 Perfect；80ms≤差值<180ms 为 Good。更早的 Bad 候选还受距离筛选；未按且晚于180ms为 Miss，不能写成对称±220ms。',
    'Hold': '头部需新按下，Perfect/Good 同 Tap，随后持续按住。断触容错按帧检查；尾部有提前220ms结算分支，但先检查当帧断触，不等于任意提前220ms松手都有效。',
    'Drag': '可用已有触点。时间差绝对值≤100ms 且满足位置条件即可记录命中，之后给 Perfect；未命中且晚于100ms为 Miss。',
    'Flick': '需要有效滑动状态，不要求每次重新按下；可沿滑动继续接 Drag。未命中且晚于140ms为 Miss；现有资料未确认提前窗口，不可推成±140ms。',
    '位置': '位置筛选看沿判定线方向的投影距离；沿法线可在屏幕平面内明显错位，不要求贴着音符图形或判定线。法向须随线旋转；仍受屏幕、遮挡/暂停区、触点候选竞争和持续接触约束。游戏局部坐标阈值不可直接当屏幕比例，也不能用手指离屏高度代替法向错位。',
    '编辑器': '80/80/100/140ms 是现有规划容差，不是新的游戏规则或 AP 保证。',
}

JUDGEMENT = {
    'source': {
        'sample': 'com.PigeonGames.Phigros-157(1).apk / arm64 libil2cpp.so',
        'sha256': 'c1a2b1fbc40b4ba93306eccaba90ac13983e15964726904aec039aff0cb657da',
        'verification': 'APK 内 libil2cpp.so 与分析文件 SHA-256 一致；静态反编译，未经设备实测',
    },
    'static_defaults_ms': {'PerfectTimeRange': 80, 'GoodTimeRange': 180, 'BadTimeRange': 220},
    'default_evidence': 'JudgeControl..cctor RVA 0x1D74010：静态字段 +4/+8/+12 = float32 0.08/0.18/0.22 秒',
    'spatial_evidence': {
        'method': 'JudgeControl.CheckNote RVA 0x1D7261C',
        'along_line_candidate_distance': 1.9,
        'early_window_narrowing_after_distance': .9,
        'early_window_formula': 'BadTimeRange + (distance - 0.9) * PerfectTimeRange * -0.5',
        'units': '游戏判定线局部坐标，不能直接当成屏幕归一化比例',
    },
    'note_rules': {
        'tap': {
            'method': 'ClickControl.Judge RVA 0x1D82114',
            'after_candidate': '已选中触点后：abs(nowTime-realTime) < PerfectTimeRange 为 Perfect；否则 < GoodTimeRange 为 Good；否则 Bad',
            'unpressed_miss': 'nowTime-realTime > GoodTimeRange；静态默认晚于 180 ms',
            'boundary': '使用严格小于；80/180 ms 边界不能当作安全落点，且仍须通过 CheckNote 的空间和触点筛选',
        },
        'hold': {
            'method': 'HoldControl.Judge RVA 0x1D83B50',
            'head': 'Perfect/Good 使用与 Tap 相同的严格时间比较，记录头部等级；按住状态沿判定线距离须 < 1.9',
            'contact_grace': '有效触点把 _safeFrame 设为 2；缺触点时先判断 <0 才 Miss，否则减1。连续缺触点的第4次 Judge 检查触发 Miss；不是固定毫秒',
            'tail': '成功按下且尚未 Miss/结算时，nowTime > realTime+holdTime-0.22 会结算；先做当次断触检查，不能保证任意提前220ms松手都有效',
        },
        'drag': {
            'method': 'DragControl.Judge RVA 0x1D828C0',
            'contact': 'abs(nowTime-realTime) <= 0.1 秒且沿判定线距离 < 2.1 可记录命中；可用持续触点',
            'score': '记录命中且 nowTime > realTime-0.005 秒后给 Perfect；未命中且晚于100ms则 Miss',
        },
        'flick': {
            'method': 'FlickControl.Judge RVA 0x1D82EC0',
            'score': '需上游 CheckFlick 设置 isJudgedForFlick；记录命中且 nowTime > realTime-0.005 秒后给 Perfect',
            'late_miss': '未完成且 nowTime-realTime > PerfectTimeRange*1.75；静态默认140ms，不能从此推导提前窗口',
            'speed_source': 'FingerManagement..ctor 0x1D7179C / Update 0x1D70B8C；flickJudgeSpeed 静态默认0.06',
            'speed_rule': '沿上次位移方向的投影/(60*deltaTime) < 0.06 或 stopped 时，当前位移长度/(60*deltaTime) >= 0.3 才置为新Flick；这是游戏坐标和帧时间归一化的速度条件，不是屏幕宽度30%',
        },
    },
    'limitations': ['静态默认值不代表挑战模式或运行时最终阈值',
        '以上是此样本的静态代码分支；帧采样、触点候选竞争和运行时字段仍影响最终结果',
        'Flick 需新滑动状态，Tap/Hold 头需新按下，持续触点可用于 Drag',
        '没有设备实测，不提供 Flick 最短距离或 AP 保证'],
    'editor_windows_ms': {'tap': 80, 'hold': 80, 'drag': 100, 'flick': 140},
    'editor_windows_note': '现有编辑器的规划容差，非上述原生字段；不能据此宣称游戏判定通过',
}

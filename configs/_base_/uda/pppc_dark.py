# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = ['pppc.py']
uda = dict(
    type='PPPCDark',
    pseudo_threshold=0.95
)

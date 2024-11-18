static uint8_t fw_cfg_read(FWCfgState *s)

{

    int arch = !!(s->cur_entry & FW_CFG_ARCH_LOCAL);

    FWCfgEntry *e = (s->cur_entry == FW_CFG_INVALID) ? NULL :

                    &s->entries[arch][s->cur_entry & FW_CFG_ENTRY_MASK];

    uint8_t ret;



    if (s->cur_entry == FW_CFG_INVALID || !e->data || s->cur_offset >= e->len)

        ret = 0;

    else {

        ret = e->data[s->cur_offset++];

    }



    trace_fw_cfg_read(s, ret);

    return ret;

}

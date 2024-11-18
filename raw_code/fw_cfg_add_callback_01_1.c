int fw_cfg_add_callback(FWCfgState *s, uint16_t key, FWCfgCallback callback,

                        void *callback_opaque, uint8_t *data, size_t len)

{

    int arch = !!(key & FW_CFG_ARCH_LOCAL);



    if (!(key & FW_CFG_WRITE_CHANNEL))

        return 0;



    key &= FW_CFG_ENTRY_MASK;



    if (key >= FW_CFG_MAX_ENTRY || len > 65535)

        return 0;



    s->entries[arch][key].data = data;

    s->entries[arch][key].len = len;

    s->entries[arch][key].callback_opaque = callback_opaque;

    s->entries[arch][key].callback = callback;



    return 1;

}

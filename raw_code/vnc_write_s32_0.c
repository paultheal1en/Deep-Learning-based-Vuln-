static void vnc_write_s32(VncState *vs, int32_t value)

{

    vnc_write_u32(vs, *(uint32_t *)&value);

}

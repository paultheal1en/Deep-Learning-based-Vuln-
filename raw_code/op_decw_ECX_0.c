void OPPROTO op_decw_ECX(void)

{

    ECX = (ECX & ~0xffff) | ((ECX - 1) & 0xffff);

}

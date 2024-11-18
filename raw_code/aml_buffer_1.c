Aml *aml_buffer(void)

{

    Aml *var = aml_bundle(0x11 /* BufferOp */, AML_BUFFER);

    return var;

}

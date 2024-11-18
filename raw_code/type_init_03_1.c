type_init(macio_register_types)



void macio_init(PCIDevice *d,

                MemoryRegion *pic_mem,

                MemoryRegion *escc_mem)

{

    MacIOState *macio_state = MACIO(d);



    macio_state->pic_mem = pic_mem;

    macio_state->escc_mem = escc_mem;

    /* Note: this code is strongly inspirated from the corresponding code

       in PearPC */





    qdev_init_nofail(DEVICE(d));

}
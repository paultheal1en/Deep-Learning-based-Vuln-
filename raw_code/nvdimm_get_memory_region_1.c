static MemoryRegion *nvdimm_get_memory_region(PCDIMMDevice *dimm)

{

    NVDIMMDevice *nvdimm = NVDIMM(dimm);



    return &nvdimm->nvdimm_mr;

}

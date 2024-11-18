libAVFilter_QueryVendorInfo(libAVFilter *this, wchar_t **info)

{

    dshowdebug("libAVFilter_QueryVendorInfo(%p)\n", this);



    if (!info)

        return E_POINTER;

    *info = wcsdup(L"libAV");



    return S_OK;

}

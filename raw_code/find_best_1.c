static unsigned int find_best(struct vf_instance *vf){

  int is_format_okay = vf->next->query_format(vf->next, IMGFMT_YV12);

  if ((is_format_okay & VFCAP_CSP_SUPPORTED_BY_HW) || (is_format_okay & VFCAP_CSP_SUPPORTED))

    return IMGFMT_YV12;

  else

    return 0;

}

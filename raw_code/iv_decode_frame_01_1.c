static unsigned long iv_decode_frame(Indeo3DecodeContext *s, 
                                     unsigned char *buf, int buf_size) 
{
  unsigned int hdr_width, hdr_height,
    chroma_width, chroma_height;
  unsigned long fflags1, fflags2, fflags3, offs1, offs2, offs3, offs;
  unsigned char *hdr_pos, *buf_pos;
  buf_pos = buf;
  buf_pos += 18;
  fflags1 = le2me_16(*(uint16_t *)buf_pos);
  buf_pos += 2;
  fflags3 = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  fflags2 = *buf_pos++;
  buf_pos += 3;
  hdr_height = le2me_16(*(uint16_t *)buf_pos);
  buf_pos += 2;
  hdr_width = le2me_16(*(uint16_t *)buf_pos);
  buf_pos += 2;
  chroma_height = ((hdr_height >> 2) + 3) & 0x7ffc;
  chroma_width = ((hdr_width >> 2) + 3) & 0x7ffc;
  offs1 = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  offs2 = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  offs3 = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 8;
  hdr_pos = buf_pos;
  if(fflags3 == 0x80) return 4;
  if(fflags1 & 0x200) {
    s->cur_frame = s->iv_frame + 1;
    s->ref_frame = s->iv_frame;
  } else {
    s->cur_frame = s->iv_frame;
    s->ref_frame = s->iv_frame + 1;
  }
  buf_pos = buf + 16 + offs1;
  offs = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  iv_Decode_Chunk(s, s->cur_frame->Ybuf, s->ref_frame->Ybuf, hdr_width, 
    hdr_height, buf_pos + offs * 2, fflags2, hdr_pos, buf_pos, 
    min(hdr_width, 160));
  if (!(s->avctx->flags & CODEC_FLAG_GRAY))
  {
  buf_pos = buf + 16 + offs2;
  offs = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  iv_Decode_Chunk(s, s->cur_frame->Vbuf, s->ref_frame->Vbuf, chroma_width, 
    chroma_height, buf_pos + offs * 2, fflags2, hdr_pos, buf_pos, 
    min(chroma_width, 40));
  buf_pos = buf + 16 + offs3;
  offs = le2me_32(*(uint32_t *)buf_pos);
  buf_pos += 4;
  iv_Decode_Chunk(s, s->cur_frame->Ubuf, s->ref_frame->Ubuf, chroma_width, 
    chroma_height, buf_pos + offs * 2, fflags2, hdr_pos, buf_pos, 
    min(chroma_width, 40));
  }
  return 8;
}
import type { AudioDTO, UploadAudioArg, UploadVideoArg, VideoDTO } from 'services/api/types';

import { api, buildV1Url } from '..';

/**
 * Builds an endpoint URL for the media router
 * @example
 * buildMediaUrl('video/upload')
 * // '/api/v1/media/video/upload'
 */
const buildMediaUrl = (path: string = '') => buildV1Url(`media/${path}`);

export const mediaApi = api.injectEndpoints({
  endpoints: (build) => ({
    uploadVideo: build.mutation<VideoDTO, UploadVideoArg>({
      query: ({ file, video_category, is_intermediate, session_id, board_id, metadata }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        return {
          url: buildMediaUrl('video/upload'),
          method: 'POST',
          body: formData,
          params: {
            video_category,
            is_intermediate,
            session_id,
            board_id,
          },
        };
      },
      invalidatesTags: ['ImageNameList'], // Invalidate list to refresh gallery
    }),
    uploadAudio: build.mutation<AudioDTO, UploadAudioArg>({
      query: ({ file, audio_category, is_intermediate, session_id, board_id, metadata }) => {
        const formData = new FormData();
        formData.append('file', file);
        if (metadata) {
          formData.append('metadata', JSON.stringify(metadata));
        }
        return {
          url: buildMediaUrl('audio/upload'),
          method: 'POST',
          body: formData,
          params: {
            audio_category,
            is_intermediate,
            session_id,
            board_id,
          },
        };
      },
      invalidatesTags: ['ImageNameList'], // Invalidate list to refresh gallery
    }),
    getVideoDTO: build.query<VideoDTO, string>({
      query: (video_name) => ({ url: buildMediaUrl(`video/${video_name}`) }),
      providesTags: (result, error, video_name) => [{ type: 'Image', id: video_name }], // Reusing Image tag for now
    }),
    getAudioDTO: build.query<AudioDTO, string>({
      query: (audio_name) => ({ url: buildMediaUrl(`audio/${audio_name}`) }),
      providesTags: (result, error, audio_name) => [{ type: 'Image', id: audio_name }], // Reusing Image tag for now
    }),
  }),
});

export const { useUploadVideoMutation, useUploadAudioMutation, useGetVideoDTOQuery, useGetAudioDTOQuery } = mediaApi;

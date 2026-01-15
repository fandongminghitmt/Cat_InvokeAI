import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { FlexProps } from '@invoke-ai/ui-library';
<<<<<<< HEAD
import { Box, Flex, Image, Text } from '@invoke-ai/ui-library';
=======
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
>>>>>>> upstream/main
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewMultipleImageState } from 'features/dnd/DndDragPreviewMultipleImage';
import { createMultipleImageDragPreview, setMultipleImageDragPreview } from 'features/dnd/DndDragPreviewMultipleImage';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { useImageContextMenu } from 'features/gallery/components/ContextMenu/ImageContextMenu';
import { GalleryItemHoverIcons } from 'features/gallery/components/ImageGrid/GalleryItemHoverIcons';
import {
  selectGetImageNamesQueryArgs,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
<<<<<<< HEAD
// import { PiImageBold, PiPlayFill, PiMusicNotesFill } from 'react-icons/pi'; // Removed to debug
=======
import { PiImageBold } from 'react-icons/pi';
>>>>>>> upstream/main
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import { galleryItemContainerSX } from './galleryItemContainerSX';

interface Props {
  imageDTO: ImageDTO;
}

const buildOnClick =
  (imageName: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const queryArgs = selectGetImageNamesQueryArgs(state);
    const imageNames = imagesApi.endpoints.getImageNames.select(queryArgs)(state).data?.image_names ?? [];

<<<<<<< HEAD
    if (imageNames.length === 0) {
=======
    // If we don't have the image names cached, we can't perform selection operations
    // This can happen if the user clicks on an image before the names are loaded
    if (imageNames.length === 0) {
      // For basic click without modifiers, we can still set selection
>>>>>>> upstream/main
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([imageName]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      if (state.gallery.imageToCompare === imageName) {
        dispatch(imageToCompareChanged(null));
      } else {
        dispatch(imageToCompareChanged(imageName));
      }
    } else if (shiftKey) {
      const rangeEndImageName = imageName;
      const lastSelectedImage = selection.at(-1);
      const lastClickedIndex = imageNames.findIndex((name) => name === lastSelectedImage);
      const currentClickedIndex = imageNames.findIndex((name) => name === rangeEndImageName);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
<<<<<<< HEAD
=======
        // We have a valid range!
>>>>>>> upstream/main
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const imagesToSelect = imageNames.slice(start, end + 1);
        if (currentClickedIndex < lastClickedIndex) {
          imagesToSelect.reverse();
        }
        dispatch(selectionChanged(uniq(selection.concat(imagesToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === imageName) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== imageName))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(imageName))));
      }
    } else {
      dispatch(selectionChanged([imageName]));
    }
  };

<<<<<<< HEAD
const isVideo = (name: string) => /\.(mp4|webm|mov)$/i.test(name);
const isAudio = (name: string) => /\.(mp3|wav|ogg)$/i.test(name);

=======
>>>>>>> upstream/main
export const GalleryImage = memo(({ imageDTO }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleImageState | DndDragPreviewMultipleImageState | null
  >(null);
  const ref = useRef<HTMLDivElement>(null);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.some((n) => n === imageDTO.image_name)),
    [imageDTO.image_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => {
          const selection = selectSelection(store.getState());
          const boardId = selectSelectedBoardId(store.getState());
<<<<<<< HEAD
=======

          // When we have multiple images selected, and the dragged image is part of the selection, initiate a
          // multi-image drag.
>>>>>>> upstream/main
          if (selection.length > 1 && selection.some((n) => n === imageDTO.image_name)) {
            return multipleImageDndSource.getData({
              image_names: selection,
              board_id: boardId,
            });
          }
<<<<<<< HEAD
          return singleImageDndSource.getData({ imageDTO }, imageDTO.image_name);
        },
        onDragStart: ({ source }) => {
=======

          // Otherwise, initiate a single-image drag
          return singleImageDndSource.getData({ imageDTO }, imageDTO.image_name);
        },
        // This is a "local" drag start event, meaning that it is only called when this specific image is dragged.
        onDragStart: ({ source }) => {
          // When we start dragging a single image, set the dragging state to true. This is only called when this
          // specific image is dragged.
>>>>>>> upstream/main
          if (singleImageDndSource.typeGuard(source.data)) {
            setIsDragging(true);
            return;
          }
        },
        onGenerateDragPreview: (args) => {
          if (multipleImageDndSource.typeGuard(args.source.data)) {
            setMultipleImageDragPreview({
              multipleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          } else if (singleImageDndSource.typeGuard(args.source.data)) {
            setSingleImageDragPreview({
              singleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          }
        },
      }),
      monitorForElements({
<<<<<<< HEAD
        onDragStart: ({ source }) => {
=======
        // This is a "global" drag start event, meaning that it is called for all drag events.
        onDragStart: ({ source }) => {
          // When we start dragging multiple images, set the dragging state to true if the dragged image is part of the
          // selection. This is called for all drag events.
>>>>>>> upstream/main
          if (
            multipleImageDndSource.typeGuard(source.data) &&
            source.data.payload.image_names.includes(imageDTO.image_name)
          ) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
<<<<<<< HEAD
=======
          // Always set the dragging state to false when a drop event occurs.
>>>>>>> upstream/main
          setIsDragging(false);
        },
      })
    );
  }, [imageDTO, store]);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useMemo(() => buildOnClick(imageDTO.image_name, store.dispatch, store.getState), [imageDTO, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    store.dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [store]);

  useImageContextMenu(imageDTO, ref);

<<<<<<< HEAD
  let content;
  if (isAudio(imageDTO.image_name)) {
    content = (
      <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center">
        <Text fontSize="xl" fontWeight="bold" color="base.500">
          Audio
        </Text>
      </Flex>
    );
  } else {
    content = (
      <>
        <Image
          pointerEvents="none"
          src={imageDTO.thumbnail_url}
          w={imageDTO.width}
          fallback={<GalleryImagePlaceholder />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        {isVideo(imageDTO.image_name) && (
          <Flex
            position="absolute"
            top={0}
            right={0}
            bottom={0}
            left={0}
            alignItems="center"
            justifyContent="center"
            pointerEvents="none"
          >
            <Box
              w={12}
              h={12}
              borderRadius="full"
              bg="whiteAlpha.600"
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <Box
                w={0}
                h={0}
                borderLeft="12px solid white"
                borderTop="8px solid transparent"
                borderBottom="8px solid transparent"
                ml={1}
              />
            </Box>
          </Flex>
        )}
      </>
    );
  }

=======
>>>>>>> upstream/main
  return (
    <>
      <Flex
        ref={ref}
        sx={galleryItemContainerSX}
        data-is-dragging={isDragging}
        data-item-id={imageDTO.image_name}
        role="button"
        onMouseOver={onMouseOver}
        onMouseOut={onMouseOut}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        data-selected={isSelected}
        data-selected-for-compare={isSelectedForCompare}
      >
<<<<<<< HEAD
        {content}
=======
        <Image
          pointerEvents="none"
          src={imageDTO.thumbnail_url}
          w={imageDTO.width}
          fallback={<GalleryImagePlaceholder />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
>>>>>>> upstream/main
        <GalleryItemHoverIcons imageDTO={imageDTO} isHovered={isHovered} />
      </Flex>
      {dragPreviewState?.type === 'multiple-image' ? createMultipleImageDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryImage.displayName = 'GalleryImage';

export const GalleryImagePlaceholder = memo((props: FlexProps) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
<<<<<<< HEAD
    {/* Removed Icon for debugging */}
    <Text>...</Text>
=======
    <Icon as={PiImageBold} boxSize={16} color="base.800" />
>>>>>>> upstream/main
  </Flex>
));

GalleryImagePlaceholder.displayName = 'GalleryImagePlaceholder';

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor, 
    EfficientNetModel, 
    RobertaModel,
    AutoModel
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import List, Optional, Tuple, Union


class ImageFeatureExtractor(nn.Module):
    r"""
    ImageFeatureExtractor uses efficientnet-b5 to extract features from images.
    """
    def __init__(self):
        super().__init__()

        self.efficient_net = EfficientNetModel.from_pretrained("google/efficientnet-b5")
        self.avg_pool2d = nn.AvgPool2d(2, ceil_mode=True)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Return a torch tensor of pooler output from final layer and flattened last hidden states.
        Shape: (Batch size, 65, 2048)
        """
        
        outputs = self.efficient_net(input)
        last_hidden_states = outputs.last_hidden_state

        # (_, 2048, 15, 15) -> (_, 2048, 8, 8) 
        pooled_hidden_states = self.avg_pool2d(last_hidden_states)
        # (_, 2048, 8, 8) -> (_, 64, 2048) 
        pooled_hidden_states = torch.flatten(pooled_hidden_states, start_dim=2).permute(0, 2, 1)

        final_model_output = outputs.pooler_output.unsqueeze(1)

        feature = torch.cat((pooled_hidden_states, final_model_output), dim=1)

        return feature


def custom_forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        image_feature_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if text_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif text_input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(text_input_ids, attention_mask)
            text_input_shape = text_input_ids.size()
        elif inputs_embeds is not None:
            text_input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = text_input_shape
        device = text_input_ids.device if text_input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + image_feature_input.size()[1] + past_key_values_length)), device=device)
        
#         print("att_mask:", attention_mask.shape)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(text_input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, text_input_shape)

#         print("extended_att_mask:", extended_attention_mask.shape)
            
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=text_input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        image_feature_input = self.embeddings.LayerNorm(image_feature_input)
        image_feature_input = self.embeddings.dropout(image_feature_input)
        embedding_output = torch.cat((embedding_output, image_feature_input), dim=1)
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class ModifiedPhoBERT(nn.Module):
    r"""
    This modified version takes input_ids and an optinal feature tensor.
    """
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.phobert.forward = custom_forward.__get__(self.phobert, RobertaModel)

    def forward(
            self,
            input_ids: torch.Tensor,
            additional_feature: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.phobert(input_ids, additional_feature)


class MultiModalClassifier(nn.Module):
    r"""
    
    """
    def __init__(self):
        super().__init__()
        self.image_feature_extractor = ImageFeatureExtractor()
        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")

        self.bridge_layer_1 = nn.Linear(2048, 768)

        self.modified_phobert_1 = ModifiedPhoBERT()

        self.pool_1 = nn.AvgPool1d(4, 4)
        self.bridge_layer_2 = nn.Linear(768, 768)

        self.modified_phobert_2 = ModifiedPhoBERT()

        self.fc1 = nn.Linear(2 * 768, 256)
        self.gelu = nn.GELU
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        image,
        ocr_text_ids,
        desc_text_ids
    ) -> torch.Tensor:
        image_tensor = self.image_processor(image, return_tensors="pt")
        image_feature = self.image_feature_extractor(image_tensor)
        image_feature = self.bridge_layer_1(image_feature)

        # phobert1
        phobert_1_output = self.modified_phobert_1(ocr_text_ids, image_feature)
        last_hidden_state_1 = phobert_1_output.last_hidden_state

        # pad the feature tensor to fixed size of (_, 256, _)
        padding_size = 256 - last_hidden_state_1.shape[1]
        last_hidden_state_1 = nn.functional.pad(
            last_hidden_state_1,
            (0, padding_size, 0, 0)
        )
        # avg pool to size (_, 32, _)
        phobert_1_feature = self.pool_1(last_hidden_state_1.permute(0, 2, 1)).permute(0, 2, 1)
        phobert_1_feature = self.bridge_layer_2(phobert_1_feature)

        # phobert2
        phobert_2_output = self.modified_phobert_2(desc_text_ids, phobert_1_feature)

        all_feature = torch.cat((phobert_1_output.pooler_output, phobert_2_output.pooler_output), dim=1)
        x = self.fc1(all_feature)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
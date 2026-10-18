# Third-party notices

This independent browser port originated in TTS Service research. The JavaScript
normalization, stress port, eager network reconstruction, and build scripts are
provided under the repository's [MIT license](LICENSE).

The following components retain their own MIT notices. Model conversion does not
replace their original licenses.

| Component | Copyright | License text |
| --- | --- | --- |
| Silero TTS `v5_cis_base` weights and reconstructed network behavior | 2025-present Silero Team | [Silero CIS MIT](licenses/silero-cis.txt) |
| Silero Stress 1.5 models, dictionaries, tokenizer, and ported behavior | 2020-present Silero Team | [Silero Stress MIT](licenses/silero-stress.txt) |
| ONNX Runtime Web 1.20.1 | Microsoft Corporation | [ONNX Runtime MIT](licenses/onnxruntime.txt) |

Upstream: [Silero Models](https://github.com/snakers4/silero-models),
[Silero Stress](https://github.com/snakers4/silero-stress), and
[ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/v1.20.1).
Silero explicitly licenses the CIS **base** models under
[LICENSE_CIS](https://github.com/snakers4/silero-models/blob/master/LICENSE_CIS);
this does not imply the same license for other Silero models.

Both builders retain these notices: the web build copies `licenses/`, and the
standalone HTML embeds the license texts in HTML comments.

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from runtime_config import (
    ConfigError,
    RuntimeConfig,
    format_config_error,
    format_config_report,
    load_env_file,
)


class RuntimeConfigTests(unittest.TestCase):
    def test_loads_typed_defaults(self):
        config = RuntimeConfig.from_environment({})

        self.assertEqual(config.transcription_backend, "whisper")
        self.assertEqual(config.osc_port, 7000)
        self.assertTrue(config.osc_control_enabled)
        self.assertEqual(config.osc_output_error_log_interval, 5.0)
        self.assertEqual(config.osc_prompt_retry_base_seconds, 0.5)
        self.assertEqual(config.osc_prompt_retry_max_seconds, 5.0)
        self.assertEqual(config.runtime_log_level, "info")
        self.assertTrue(config.runtime_log_console_enabled)
        self.assertEqual(config.runtime_log_file, "")
        self.assertEqual(config.runtime_shutdown_grace_seconds, 25.0)
        self.assertEqual(
            config.transcription_final_overflow_policy,
            "drop_oldest",
        )
        self.assertEqual(config.audio_input_device_index, None)
        self.assertEqual(config.default_gender, "neutral")
        self.assertEqual(config.default_age, "adult")
        self.assertEqual(config.default_visual_mode, "asian_american")
        self.assertEqual(config.default_prompt_style, "human_focus")
        self.assertEqual(config.default_language, "auto")
        self.assertEqual(
            config.prompt_tokenizer_models,
            (
                "openai/clip-vit-large-patch14",
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            ),
        )
        self.assertEqual(
            config.prompt_token_budget_fallback,
            "conservative",
        )
        self.assertEqual(
            config.prompt_refinement_chain,
            (
                "ollama:gemini-3-flash-preview",
                "ollama:kimi-k2.6:cloud",
                "ollama:gemma4:31b-cloud",
            ),
        )
        self.assertEqual(config.prompt_refinement_request_timeout, 15.0)
        self.assertEqual(config.prompt_refinement_target_tokens, 70)
        self.assertEqual(config.prompt_refinement_max_output_characters, 1000)

    def test_every_setting_has_a_unique_environment_name(self):
        environment_names = [
            config_field.metadata["env"]
            for config_field in fields(RuntimeConfig)
        ]

        self.assertEqual(len(environment_names), len(set(environment_names)))

    def test_loads_and_validates_prompt_retry_intervals(self):
        config = RuntimeConfig.from_environment({
            "OSC_PROMPT_RETRY_BASE_SECONDS": "0.25",
            "OSC_PROMPT_RETRY_MAX_SECONDS": "2.0",
        })
        self.assertEqual(config.osc_prompt_retry_base_seconds, 0.25)
        self.assertEqual(config.osc_prompt_retry_max_seconds, 2.0)
        for name in ("OSC_PROMPT_RETRY_BASE_SECONDS", "OSC_PROMPT_RETRY_MAX_SECONDS"):
            for value in ("0", "-1", "nan", "inf", "not-a-number"):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigError):
                    RuntimeConfig.from_environment({name: value})
        with self.assertRaises(ConfigError) as context:
            RuntimeConfig.from_environment({
                "OSC_PROMPT_RETRY_BASE_SECONDS": "6",
                "OSC_PROMPT_RETRY_MAX_SECONDS": "5",
            })
        self.assertIn(
            "OSC_PROMPT_RETRY_BASE_SECONDS must not exceed OSC_PROMPT_RETRY_MAX_SECONDS",
            context.exception.errors,
        )

    def test_command_line_overrides_take_precedence(self):
        config = RuntimeConfig.from_environment(
            {
                "TRANSCRIPTION_BACKEND": "whisper",
                "AUDIO_INPUT_DEVICE_INDEX": "2",
                "GROQ_API_KEY": "test-key",
                "DEFAULT_GENDER": "neutral",
                "DEFAULT_AGE": "adult",
                "DEFAULT_VISUAL_MODE": "asian_american",
                "DEFAULT_PROMPT_STYLE": "human_focus",
                "DEFAULT_LANGUAGE": "auto",
            },
            backend_override="groq",
            input_device_override=7,
            gender_override="woman",
            age_override="elder",
            visual_mode_override="black_brown",
            prompt_style_override="general_scene",
            language_override="zh",
        )

        self.assertEqual(config.transcription_backend, "groq")
        self.assertEqual(config.audio_input_device_index, 7)
        self.assertEqual(config.default_gender, "woman")
        self.assertEqual(config.default_age, "elder")
        self.assertEqual(config.default_visual_mode, "black_brown")
        self.assertEqual(config.default_prompt_style, "general_scene")
        self.assertEqual(config.default_language, "zh")

    def test_normalizes_startup_control_environment_values(self):
        config = RuntimeConfig.from_environment(
            {
                "DEFAULT_GENDER": "WOMAN",
                "DEFAULT_AGE": "ELDER",
                "DEFAULT_VISUAL_MODE": "BLACK_BROWN",
                "DEFAULT_PROMPT_STYLE": "GENERAL_SCENE",
                "DEFAULT_LANGUAGE": "ES",
            }
        )

        self.assertEqual(config.default_gender, "woman")
        self.assertEqual(config.default_age, "elder")
        self.assertEqual(config.default_visual_mode, "black_brown")
        self.assertEqual(config.default_prompt_style, "general_scene")
        self.assertEqual(config.default_language, "es")

    def test_normalizes_prompt_budget_fallback(self):
        config = RuntimeConfig.from_environment(
            {"PROMPT_TOKEN_BUDGET_FALLBACK": "OFF"}
        )

        self.assertEqual(config.prompt_token_budget_fallback, "off")

    def test_normalizes_final_overflow_policy(self):
        config = RuntimeConfig.from_environment(
            {"TRANSCRIPTION_FINAL_OVERFLOW_POLICY": "DROP_NEWEST"}
        )

        self.assertEqual(
            config.transcription_final_overflow_policy,
            "drop_newest",
        )

    def test_rejects_invalid_startup_controls_together(self):
        with self.assertRaises(ConfigError) as context:
            RuntimeConfig.from_environment(
                {
                    "DEFAULT_GENDER": "robot",
                    "DEFAULT_AGE": "ancient",
                    "DEFAULT_VISUAL_MODE": "unknown",
                    "DEFAULT_PROMPT_STYLE": "abstract",
                    "DEFAULT_LANGUAGE": "fr",
                }
            )

        errors = context.exception.errors
        self.assertIn(
            "DEFAULT_GENDER must be one of: man, neutral, woman",
            errors,
        )
        self.assertIn(
            "DEFAULT_AGE must be one of: adult, elder, young",
            errors,
        )
        self.assertIn(
            "DEFAULT_VISUAL_MODE must be one of: "
            "asian_american, asian_black_brown, black_brown",
            errors,
        )
        self.assertIn(
            "DEFAULT_PROMPT_STYLE must be one of: general_scene, human_focus",
            errors,
        )
        self.assertIn(
            "DEFAULT_LANGUAGE must be one of: auto, en, es, zh",
            errors,
        )

    def test_rejects_malformed_numbers_and_booleans(self):
        with self.assertRaises(ConfigError) as context:
            RuntimeConfig.from_environment(
                {
                    "OSC_PORT": "seven-thousand",
                    "OSC_CONTROL_ENABLED": "sometimes",
                }
            )

        self.assertIn("OSC_PORT must be an integer", context.exception.errors)
        self.assertTrue(
            any(
                message.startswith("OSC_CONTROL_ENABLED must be one of:")
                for message in context.exception.errors
            )
        )

    def test_rejects_invalid_ranges_and_cross_field_values(self):
        with self.assertRaises(ConfigError) as context:
            RuntimeConfig.from_environment(
                {
                    "OSC_PORT": "70000",
                    "VAD_THRESHOLD": "1.2",
                    "WHISPER_MIN_AUDIO_SECONDS": "8",
                    "WHISPER_MAX_AUDIO_SECONDS": "4",
                    "TRANSCRIPTION_RETRY_BASE_SECONDS": "12",
                    "TRANSCRIPTION_RETRY_MAX_SECONDS": "3",
                    "RUNTIME_LOG_LEVEL": "verbose",
                    "RUNTIME_LOG_MAX_BYTES": "0",
                    "RUNTIME_LOG_BACKUP_COUNT": "-1",
                    "RUNTIME_SHUTDOWN_GRACE_SECONDS": "0",
                    "OSC_OUTPUT_ERROR_LOG_INTERVAL": "0",
                    "PROMPT_TOKEN_BUDGET_FALLBACK": "approximate",
                    "TRANSCRIPTION_FINAL_OVERFLOW_POLICY": "random",
                    "PROMPT_MAX_TOKENS": "1",
                }
            )

        errors = context.exception.errors
        self.assertIn("OSC_PORT must be between 1 and 65535", errors)
        self.assertIn("VAD_THRESHOLD must be between 0 and 1", errors)
        self.assertIn(
            "WHISPER_MIN_AUDIO_SECONDS must not exceed WHISPER_MAX_AUDIO_SECONDS",
            errors,
        )
        self.assertIn(
            "TRANSCRIPTION_RETRY_BASE_SECONDS must not exceed "
            "TRANSCRIPTION_RETRY_MAX_SECONDS",
            errors,
        )
        self.assertIn(
            "RUNTIME_LOG_LEVEL must be one of: "
            "critical, debug, error, info, warning",
            errors,
        )
        self.assertIn(
            "RUNTIME_LOG_MAX_BYTES must be greater than 0",
            errors,
        )
        self.assertIn(
            "RUNTIME_LOG_BACKUP_COUNT must be greater than 0",
            errors,
        )
        self.assertIn(
            "RUNTIME_SHUTDOWN_GRACE_SECONDS must be greater than 0",
            errors,
        )
        self.assertIn(
            "OSC_OUTPUT_ERROR_LOG_INTERVAL must be greater than 0",
            errors,
        )
        self.assertIn(
            "PROMPT_TOKEN_BUDGET_FALLBACK must be one of: conservative, off",
            errors,
        )
        self.assertIn(
            "TRANSCRIPTION_FINAL_OVERFLOW_POLICY must be one of: "
            "drop_newest, drop_oldest",
            errors,
        )
        self.assertIn("PROMPT_MAX_TOKENS must be at least 2", errors)

    def test_requires_a_real_groq_key_for_a_groq_backend(self):
        for key in ("", "your_groq_key_here"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ConfigError,
                    "TRANSCRIPTION_BACKEND=groq requires GROQ_API_KEY",
                ):
                    RuntimeConfig.from_environment(
                        {
                            "TRANSCRIPTION_BACKEND": "groq",
                            "GROQ_API_KEY": key,
                        }
                    )

    def test_parses_a_configurable_prompt_refinement_chain(self):
        config = RuntimeConfig.from_environment(
            {
                "PROMPT_REFINEMENT_CHAIN": (
                    "OLLAMA:local-model,capriole:hosted:model"
                ),
                "CAPRIOLE_API_KEY": "capriole-secret",
                "PROMPT_REFINEMENT_REQUEST_TIMEOUT": "6.5",
                "PROMPT_REFINEMENT_TARGET_TOKENS": "64",
                "PROMPT_REFINEMENT_MAX_OUTPUT_CHARACTERS": "900",
            }
        )

        self.assertEqual(
            config.prompt_refinement_chain,
            ("OLLAMA:local-model", "capriole:hosted:model"),
        )
        self.assertEqual(config.prompt_refinement_request_timeout, 6.5)
        self.assertEqual(config.prompt_refinement_target_tokens, 64)
        self.assertEqual(config.prompt_refinement_max_output_characters, 900)

    def test_validates_prompt_refinement_routes_and_limits_together(self):
        with self.assertRaises(ConfigError) as context:
            RuntimeConfig.from_environment(
                {
                    "PROMPT_REFINEMENT_CHAIN": "missing-route,other:model",
                    "PROMPT_REFINEMENT_OLLAMA_ENDPOINT": "localhost:11434",
                    "PROMPT_REFINEMENT_CAPRIOLE_ENDPOINT": "ftp://example.test",
                    "PROMPT_REFINEMENT_REQUEST_TIMEOUT": "0",
                    "PROMPT_REFINEMENT_TARGET_TOKENS": "-1",
                    "PROMPT_REFINEMENT_MAX_OUTPUT_CHARACTERS": "0",
                }
            )

        errors = context.exception.errors
        self.assertIn(
            "PROMPT_REFINEMENT_CHAIN entries must use provider:model",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_CHAIN provider must be one of: capriole, ollama",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_OLLAMA_ENDPOINT must be an absolute HTTP(S) URL",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_CAPRIOLE_ENDPOINT must be an absolute HTTP(S) URL",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_REQUEST_TIMEOUT must be greater than 0",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_TARGET_TOKENS must be greater than 0",
            errors,
        )
        self.assertIn(
            "PROMPT_REFINEMENT_MAX_OUTPUT_CHARACTERS must be greater than 0",
            errors,
        )

    def test_capriole_refinement_route_requires_a_real_key(self):
        for key in ("", "your_capriole_key_here"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    ConfigError,
                    "PROMPT_REFINEMENT_CHAIN with capriole requires "
                    "CAPRIOLE_API_KEY",
                ):
                    RuntimeConfig.from_environment(
                        {
                            "PROMPT_REFINEMENT_CHAIN": "capriole:model",
                            "CAPRIOLE_API_KEY": key,
                        }
                    )

    def test_preserves_legacy_online_timing_fallbacks(self):
        config = RuntimeConfig.from_environment(
            {
                "ONLINE_TRANSCRIPTION_INTERVAL": "4.5",
                "ONLINE_MIN_AUDIO_SECONDS": "2.25",
            }
        )

        self.assertEqual(config.groq_transcription_interval, 4.5)
        self.assertEqual(config.google_transcription_interval, 4.5)
        self.assertEqual(config.groq_min_audio_seconds, 2.25)
        self.assertEqual(config.google_min_audio_seconds, 2.25)

    def test_report_redacts_secrets(self):
        config = RuntimeConfig.from_environment(
            {
                "CAPRIOLE_API_KEY": "capriole-secret",
                "GROQ_API_KEY": "groq-secret",
            }
        )

        report = format_config_report(config)

        self.assertNotIn("capriole-secret", report)
        self.assertNotIn("groq-secret", report)
        self.assertIn("CAPRIOLE_API_KEY", report)
        self.assertIn("GROQ_API_KEY", report)
        self.assertGreaterEqual(report.count("<redacted>"), 2)

    def test_report_marks_an_empty_runtime_log_file_as_disabled(self):
        report = format_config_report(RuntimeConfig())

        self.assertIn("RUNTIME_LOG_FILE", report)
        self.assertIn("<disabled>", report)

    def test_error_report_never_includes_unrelated_environment_values(self):
        try:
            RuntimeConfig.from_environment({"OSC_PORT": "invalid"})
        except ConfigError as error:
            report = format_config_error(error)
        else:
            self.fail("Expected invalid configuration")

        self.assertIn("OSC_PORT must be an integer", report)
        self.assertNotIn("invalid", report)


class EnvironmentFileTests(unittest.TestCase):
    def test_loads_values_without_overwriting_the_process_environment(self):
        environment = {"OSC_PORT": "9000"}
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# local values\nOSC_PORT=7000\nGROQ_API_KEY='secret'\n",
                encoding="utf-8",
            )

            load_env_file(env_path, environment)

        self.assertEqual(environment["OSC_PORT"], "9000")
        self.assertEqual(environment["GROQ_API_KEY"], "secret")


class ConfigCommandTests(unittest.TestCase):
    @staticmethod
    def clean_config_environment():
        environment = os.environ.copy()
        defaults = RuntimeConfig()
        for config_field in fields(defaults):
            env_name = config_field.metadata["env"]
            value = getattr(defaults, config_field.name)
            if value is None:
                display = ""
            elif isinstance(value, bool):
                display = str(value).lower()
            elif isinstance(value, tuple):
                display = ",".join(str(item) for item in value)
            else:
                display = str(value)
            environment[env_name] = display
        environment["TRANSCRIPTION_BACKEND"] = "google"
        return environment

    def run_config_check(self, environment):
        return subprocess.run(
            [sys.executable, "transcriber.py", "--check-config"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_importing_transcriber_does_not_load_or_validate_runtime_config(self):
        environment = self.clean_config_environment()
        environment["OSC_PORT"] = "invalid"
        environment.pop("TRANSFORMERS_VERBOSITY", None)
        environment.pop("HF_HUB_DISABLE_SYMLINKS_WARNING", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; import transcriber; "
                    "assert 'TRANSFORMERS_VERBOSITY' not in os.environ; "
                    "assert 'HF_HUB_DISABLE_SYMLINKS_WARNING' not in os.environ; "
                    "print('imported')"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "imported")

    def test_check_config_returns_two_for_invalid_values(self):
        environment = self.clean_config_environment()
        environment["OSC_PORT"] = "invalid"

        result = self.run_config_check(environment)

        self.assertEqual(result.returncode, 2)
        self.assertIn("OSC_PORT must be an integer", result.stdout)

    def test_check_config_redacts_secret_values(self):
        environment = self.clean_config_environment()
        environment["GROQ_API_KEY"] = "groq-test-secret"
        environment["CAPRIOLE_API_KEY"] = "capriole-test-secret"

        result = self.run_config_check(environment)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("groq-test-secret", result.stdout)
        self.assertNotIn("capriole-test-secret", result.stdout)
        self.assertGreaterEqual(result.stdout.count("<redacted>"), 2)


if __name__ == "__main__":
    unittest.main()

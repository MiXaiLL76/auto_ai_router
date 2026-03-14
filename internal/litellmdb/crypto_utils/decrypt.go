package cryptoutils

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"golang.org/x/crypto/nacl/secretbox"
)

func DecryptValue(value []byte, signingKey string) (string, error) {
	if len(value) == 0 {
		return "", nil
	}

	// 1. Получаем 32-байтный ключ через SHA-256 (аналог hashlib.sha256)
	hash := sha256.Sum256([]byte(signingKey))
	var secretKey [32]byte
	copy(secretKey[:], hash[:])

	// 2. В NaCl SecretBox первые 24 байта зашифрованного сообщения — это Nonce (одноразовый код)
	if len(value) < 24 {
		return "", fmt.Errorf("ciphertext too short")
	}

	var nonce [24]byte
	copy(nonce[:], value[:24])
	ciphertext := value[24:]

	// 3. Дешифровка
	// Open расшифровывает и проверяет аутентичность (Poly1305)
	opened, ok := secretbox.Open(nil, ciphertext, &nonce, &secretKey)
	if !ok {
		return "", fmt.Errorf("decryption failed (invalid key or tampered data)")
	}
	return string(opened), nil
}

func DecryptValueHelper(value string, key string, signingKey string) (string, error) {
	// Попытка декодирования Base64 (сначала URL-safe, затем Standard)
	var decoded []byte
	var err error

	decoded, err = base64.URLEncoding.DecodeString(value)
	if err != nil {
		// Если не вышло URL-safe, пробуем стандартный (аналог fallback в Python)
		decoded, err = base64.StdEncoding.DecodeString(value)
		if err != nil {
			return value, fmt.Errorf("base64 decode error: %v", err)
		}
	}

	// Вызов основной функции дешифровки
	decrypted, err := DecryptValue(decoded, signingKey)
	if err != nil {
		return "", fmt.Errorf("decrypting %s: %w", key, err)
	}

	return decrypted, nil
}

func DecryptCredentialLiteLLMParams(params *queries.CredentialLiteLLMParams, signingKey string) error {
	if params == nil {
		return nil
	}

	decryptField := func(fieldName string, ptr **string) error {
		if *ptr == nil || **ptr == "" {
			return nil
		}
		decrypted, err := DecryptValueHelper(**ptr, fieldName, signingKey)
		if err != nil {
			return err
		}
		*ptr = &decrypted
		return nil
	}

	if err := decryptField("api_base", &params.APIBase); err != nil {
		return err
	}
	if err := decryptField("api_key", &params.APIKey); err != nil {
		return err
	}
	if err := decryptField("vertex_project", &params.VertexProject); err != nil {
		return err
	}
	if err := decryptField("vertex_location", &params.VertexLocation); err != nil {
		return err
	}
	if err := decryptField("vertex_credentials", &params.VertexCredentials); err != nil {
		return err
	}
	if err := decryptField("custom_llm_provider", &params.CustomLLMProviderName); err != nil {
		return err
	}
	if err := decryptField("model", &params.Model); err != nil {
		return err
	}

	return nil
}

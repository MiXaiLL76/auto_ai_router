package spendlog

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
)

// SpendUpdates - агрегированные обновления spend с явной типизацией
// Позволяет избежать путаницы со строковыми ключами и повышает читаемость кода
type SpendUpdates struct {
	Tokens      map[string]float64 // apiKey -> amount
	Users       map[string]float64 // userID -> amount
	Teams       map[string]float64 // teamID -> amount
	Orgs        map[string]float64 // orgID -> amount
	TeamMembers map[string]float64 // "teamID:userID" -> amount
	OrgMembers  map[string]float64 // "orgID:userID" -> amount
}

// aggregateSpendUpdates группирует обновления spend по сущностям
// Вместо N UPDATE запросов для N SpendLogEntry, агрегирует их в ~5-10 операций
//
// Пример:
//
//	Батч из 100 запросов:
//	  - APIKey: "abc", Spend: 10
//	  - APIKey: "abc", Spend: 5    ← одна сущность!
//	  - UserID: "user1", Spend: 8
//	  - UserID: "user1", Spend: 7  ← одна сущность!
//
//	Результат:
//	  SpendUpdates {
//	    Tokens: {"abc": 15},       ← 1 UPDATE вместо 2
//	    Users: {"user1": 15},      ← 1 UPDATE вместо 2
//	  }
func aggregateSpendUpdates(batch []*models.SpendLogEntry) *SpendUpdates {
	updates := &SpendUpdates{
		Tokens:      make(map[string]float64),
		Users:       make(map[string]float64),
		Teams:       make(map[string]float64),
		Orgs:        make(map[string]float64),
		TeamMembers: make(map[string]float64),
		OrgMembers:  make(map[string]float64),
	}

	for _, entry := range batch {
		// Token (всегда)
		updates.Tokens[entry.APIKey] += entry.Spend

		// User (если есть)
		if entry.UserID != "" {
			updates.Users[entry.UserID] += entry.Spend
		}

		// Team (если есть)
		if entry.TeamID != "" {
			updates.Teams[entry.TeamID] += entry.Spend
		}

		// Organization (если есть)
		if entry.OrganizationID != "" {
			updates.Orgs[entry.OrganizationID] += entry.Spend
		}

		// TeamMembership (если User + Team)
		if entry.UserID != "" && entry.TeamID != "" {
			key := fmt.Sprintf("%s:%s", entry.TeamID, entry.UserID)
			updates.TeamMembers[key] += entry.Spend
		}

		// OrganizationMembership (если User + Org)
		if entry.UserID != "" && entry.OrganizationID != "" {
			key := fmt.Sprintf("%s:%s", entry.OrganizationID, entry.UserID)
			updates.OrgMembers[key] += entry.Spend
		}
	}

	return updates
}

// executeSpendUpdates выполняет все UPDATE операции в одной транзакции
// Если какая-то операция упадёт, вся транзакция откатится (atomicity)
func executeSpendUpdates(ctx context.Context, tx pgx.Tx, updates *SpendUpdates) error {
	if updates == nil {
		return nil
	}

	// Выполняем каждый тип обновления (пропускаем пустые карты)
	if len(updates.Tokens) > 0 {
		if err := updateTokens(ctx, tx, updates.Tokens); err != nil {
			return fmt.Errorf("update tokens: %w", err)
		}
	}
	if len(updates.Users) > 0 {
		if err := updateUsers(ctx, tx, updates.Users); err != nil {
			return fmt.Errorf("update users: %w", err)
		}
	}
	if len(updates.Teams) > 0 {
		if err := updateTeams(ctx, tx, updates.Teams); err != nil {
			return fmt.Errorf("update teams: %w", err)
		}
	}
	if len(updates.Orgs) > 0 {
		if err := updateOrgs(ctx, tx, updates.Orgs); err != nil {
			return fmt.Errorf("update orgs: %w", err)
		}
	}
	if len(updates.TeamMembers) > 0 {
		if err := updateTeamMembers(ctx, tx, updates.TeamMembers); err != nil {
			return fmt.Errorf("update team members: %w", err)
		}
	}
	if len(updates.OrgMembers) > 0 {
		if err := updateOrgMembers(ctx, tx, updates.OrgMembers); err != nil {
			return fmt.Errorf("update org members: %w", err)
		}
	}

	return nil
}

// updateTokens - обновить Token.spend в LiteLLM_VerificationToken
// Группирует обновления по apiKey и выполняет одинарный UPDATE для каждого ключа
func updateTokens(ctx context.Context, tx pgx.Tx, tokens map[string]float64) error {
	for apiKey, amount := range tokens {
		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_VerificationToken" SET spend = spend + $1 WHERE token = $2 AND spend IS NOT NULL`,
			amount, apiKey)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateUsers - обновить LiteLLM_UserTable.spend
// Проверяет spend IS NOT NULL для избежания случайного обновления null значений
func updateUsers(ctx context.Context, tx pgx.Tx, users map[string]float64) error {
	for userID, amount := range users {
		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_UserTable" SET spend = spend + $1 WHERE user_id = $2 AND spend IS NOT NULL`,
			amount, userID)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateTeams - обновить LiteLLM_TeamTable.spend
// Проверяет spend IS NOT NULL для избежания случайного обновления null значений
func updateTeams(ctx context.Context, tx pgx.Tx, teams map[string]float64) error {
	for teamID, amount := range teams {
		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_TeamTable" SET spend = spend + $1 WHERE team_id = $2 AND spend IS NOT NULL`,
			amount, teamID)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateOrgs - обновить LiteLLM_OrganizationTable.spend
// Проверяет spend IS NOT NULL для избежания случайного обновления null значений
func updateOrgs(ctx context.Context, tx pgx.Tx, orgs map[string]float64) error {
	for orgID, amount := range orgs {
		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_OrganizationTable" SET spend = spend + $1 WHERE organization_id = $2 AND spend IS NOT NULL`,
			amount, orgID)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateTeamMembers - обновить LiteLLM_TeamMembership.spend
// Используется ключ "teamID:userID" для идентификации записи
// Проверяет spend IS NOT NULL для избежания случайного обновления null значений
func updateTeamMembers(ctx context.Context, tx pgx.Tx, teamMembers map[string]float64) error {
	for key, amount := range teamMembers {
		// key формата "teamID:userID"
		parts := strings.SplitN(key, ":", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid team member key format %q: expected 'teamID:userID'", key)
		}
		teamID := parts[0]
		userID := parts[1]

		if teamID == "" || userID == "" {
			return fmt.Errorf("invalid team member key %q: empty teamID or userID", key)
		}

		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_TeamMembership" SET spend = spend + $1 WHERE team_id = $2 AND user_id = $3 AND spend IS NOT NULL`,
			amount, teamID, userID)
		if err != nil {
			return err
		}
	}
	return nil
}

// updateOrgMembers - обновить LiteLLM_OrganizationMembership.spend
// Используется ключ "orgID:userID" для идентификации записи
// Проверяет spend IS NOT NULL для избежания случайного обновления null значений
func updateOrgMembers(ctx context.Context, tx pgx.Tx, orgMembers map[string]float64) error {
	for key, amount := range orgMembers {
		// key формата "orgID:userID"
		parts := strings.SplitN(key, ":", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid org member key format %q: expected 'orgID:userID'", key)
		}
		orgID := parts[0]
		userID := parts[1]

		if orgID == "" || userID == "" {
			return fmt.Errorf("invalid org member key %q: empty orgID or userID", key)
		}

		_, err := tx.Exec(ctx,
			`UPDATE "LiteLLM_OrganizationMembership" SET spend = spend + $1 WHERE organization_id = $2 AND user_id = $3 AND spend IS NOT NULL`,
			amount, orgID, userID)
		if err != nil {
			return err
		}
	}
	return nil
}
